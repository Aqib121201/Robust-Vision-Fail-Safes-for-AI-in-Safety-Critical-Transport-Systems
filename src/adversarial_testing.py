"""
Adversarial testing module for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Implements various adversarial attacks and robustness testing procedures.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import imagenet_utils
import cv2
import logging
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

from .config import (
    ADVERSARIAL_CONFIG, VISUALIZATIONS_DIR, LOGGING_CONFIG,
    VEHICLE_CLASSES, SAFETY_CRITICAL_CLASSES
)

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class AdversarialTester:
    """
    Adversarial testing class for evaluating model robustness.
    Implements various noise injection and occlusion techniques.
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any] = None):
        """
        Initialize the adversarial tester.
        
        Args:
            model: Trained Keras model to test
            config: Configuration dictionary for adversarial testing
        """
        self.model = model
        self.config = config or ADVERSARIAL_CONFIG
        self.results = {}
        
        logger.info("AdversarialTester initialized")
    
    def add_gaussian_noise(self, images: np.ndarray, std: float) -> np.ndarray:
        """
        Add Gaussian noise to images.
        
        Args:
            images: Input images
            std: Standard deviation of noise
            
        Returns:
            Images with added noise
        """
        noise = np.random.normal(0, std, images.shape)
        noisy_images = np.clip(images + noise, 0, 1)
        return noisy_images
    
    def add_salt_pepper_noise(self, images: np.ndarray, prob: float) -> np.ndarray:
        """
        Add salt and pepper noise to images.
        
        Args:
            images: Input images
            prob: Probability of noise
            
        Returns:
            Images with salt and pepper noise
        """
        noisy_images = images.copy()
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(images.shape) < prob / 2
        noisy_images[salt_mask] = 1
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(images.shape) < prob / 2
        noisy_images[pepper_mask] = 0
        
        return noisy_images
    
    def add_occlusion(self, images: np.ndarray, occlusion_size: float) -> np.ndarray:
        """
        Add random occlusion to images.
        
        Args:
            images: Input images
            occlusion_size: Fraction of image to occlude
            
        Returns:
            Images with random occlusions
        """
        occluded_images = images.copy()
        h, w = images.shape[1:3]
        
        for i in range(len(images)):
            # Random occlusion position
            occlude_h = int(h * occlusion_size)
            occlude_w = int(w * occlusion_size)
            
            start_h = np.random.randint(0, h - occlude_h + 1)
            start_w = np.random.randint(0, w - occlude_w + 1)
            
            # Apply occlusion (black rectangle)
            occluded_images[i, start_h:start_h+occlude_h, start_w:start_w+occlude_w, :] = 0
        
        return occluded_images
    
    def add_blur(self, images: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Add Gaussian blur to images.
        
        Args:
            images: Input images
            kernel_size: Size of blur kernel
            
        Returns:
            Blurred images
        """
        blurred_images = np.zeros_like(images)
        
        for i in range(len(images)):
            # Convert to uint8 for OpenCV
            img_uint8 = (images[i] * 255).astype(np.uint8)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
            
            # Convert back to float
            blurred_images[i] = blurred.astype(np.float32) / 255.0
        
        return blurred_images
    
    def fgsm_attack(self, images: np.ndarray, labels: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Attack strength
            
        Returns:
            Adversarial images
        """
        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = self.model(images)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        
        gradients = tape.gradient(loss, images)
        perturbations = epsilon * tf.sign(gradients)
        adversarial_images = tf.clip_by_value(images + perturbations, 0, 1)
        
        return adversarial_images.numpy()
    
    def pgd_attack(self, images: np.ndarray, labels: np.ndarray, 
                   epsilon: float, alpha: float, steps: int) -> np.ndarray:
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            images: Input images
            labels: True labels
            epsilon: Maximum perturbation
            alpha: Step size
            steps: Number of steps
            
        Returns:
            Adversarial images
        """
        adversarial_images = images.copy()
        
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_images)
                predictions = self.model(adversarial_images)
                loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            
            gradients = tape.gradient(loss, adversarial_images)
            perturbations = alpha * tf.sign(gradients)
            adversarial_images = tf.clip_by_value(adversarial_images + perturbations, 
                                                images - epsilon, images + epsilon)
            adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)
        
        return adversarial_images.numpy()
    
    def test_noise_robustness(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Test model robustness against various types of noise.
        
        Args:
            images: Test images
            labels: True labels
            
        Returns:
            Dictionary with robustness results
        """
        logger.info("Testing noise robustness...")
        
        results = {
            'gaussian_noise': {},
            'salt_pepper_noise': {},
            'occlusion': {},
            'blur': {}
        }
        
        # Test Gaussian noise
        for std in self.config['noise_intensities']:
            noisy_images = self.add_gaussian_noise(images, std)
            predictions = self.model.predict(noisy_images)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            results['gaussian_noise'][f'std_{std}'] = accuracy
        
        # Test salt and pepper noise
        for prob in self.config['noise_intensities']:
            noisy_images = self.add_salt_pepper_noise(images, prob)
            predictions = self.model.predict(noisy_images)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            results['salt_pepper_noise'][f'prob_{prob}'] = accuracy
        
        # Test occlusion
        for size in self.config['occlusion_sizes']:
            occluded_images = self.add_occlusion(images, size)
            predictions = self.model.predict(occluded_images)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            results['occlusion'][f'size_{size}'] = accuracy
        
        # Test blur
        for kernel_size in self.config['blur_kernels']:
            blurred_images = self.add_blur(images, kernel_size)
            predictions = self.model.predict(blurred_images)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            results['blur'][f'kernel_{kernel_size}'] = accuracy
        
        self.results['noise_robustness'] = results
        logger.info("Noise robustness testing completed")
        
        return results
    
    def test_adversarial_attacks(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            images: Test images
            labels: True labels
            
        Returns:
            Dictionary with adversarial attack results
        """
        logger.info("Testing adversarial attacks...")
        
        results = {
            'fgsm': {},
            'pgd': {}
        }
        
        # Test FGSM attacks
        for epsilon in self.config['fgsm_epsilon']:
            adversarial_images = self.fgsm_attack(images, labels, epsilon)
            predictions = self.model.predict(adversarial_images)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            results['fgsm'][f'epsilon_{epsilon}'] = accuracy
        
        # Test PGD attacks
        adversarial_images = self.pgd_attack(
            images, labels, 
            self.config['pgd_epsilon'],
            self.config['pgd_alpha'],
            self.config['pgd_steps']
        )
        predictions = self.model.predict(adversarial_images)
        accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
        results['pgd']['default'] = accuracy
        
        self.results['adversarial_attacks'] = results
        logger.info("Adversarial attack testing completed")
        
        return results
    
    def analyze_confidence_drops(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze confidence drops under adversarial conditions.
        
        Args:
            images: Test images
            labels: True labels
            
        Returns:
            Dictionary with confidence analysis
        """
        logger.info("Analyzing confidence drops...")
        
        # Get original predictions
        original_predictions = self.model.predict(images)
        original_confidences = np.max(original_predictions, axis=1)
        original_accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(original_predictions, axis=1))
        
        confidence_analysis = {
            'original': {
                'accuracy': original_accuracy,
                'avg_confidence': np.mean(original_confidences),
                'min_confidence': np.min(original_confidences),
                'confidence_std': np.std(original_confidences)
            }
        }
        
        # Test with different noise levels
        for std in [0.1, 0.2, 0.3]:
            noisy_images = self.add_gaussian_noise(images, std)
            predictions = self.model.predict(noisy_images)
            confidences = np.max(predictions, axis=1)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            
            confidence_analysis[f'gaussian_noise_{std}'] = {
                'accuracy': accuracy,
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'confidence_std': np.std(confidences),
                'confidence_drop': np.mean(original_confidences - confidences)
            }
        
        # Test with occlusion
        for size in [0.2, 0.4]:
            occluded_images = self.add_occlusion(images, size)
            predictions = self.model.predict(occluded_images)
            confidences = np.max(predictions, axis=1)
            accuracy = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
            
            confidence_analysis[f'occlusion_{size}'] = {
                'accuracy': accuracy,
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'confidence_std': np.std(confidences),
                'confidence_drop': np.mean(original_confidences - confidences)
            }
        
        self.results['confidence_analysis'] = confidence_analysis
        logger.info("Confidence analysis completed")
        
        return confidence_analysis
    
    def visualize_attacks(self, images: np.ndarray, labels: np.ndarray, 
                         num_samples: int = 5) -> None:
        """
        Visualize adversarial attacks and their effects.
        
        Args:
            images: Test images
            labels: True labels
            num_samples: Number of samples to visualize
        """
        logger.info("Creating attack visualizations...")
        
        # Select random samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        sample_images = images[indices]
        sample_labels = labels[indices]
        
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
        
        for i, idx in enumerate(indices):
            # Original image
            axes[i, 0].imshow(sample_images[i])
            axes[i, 0].set_title(f'Original\n{VEHICLE_CLASSES[np.argmax(sample_labels[i])]}')
            axes[i, 0].axis('off')
            
            # Gaussian noise
            noisy_img = self.add_gaussian_noise(sample_images[i:i+1], 0.2)[0]
            axes[i, 1].imshow(noisy_img)
            axes[i, 1].set_title('Gaussian Noise')
            axes[i, 1].axis('off')
            
            # Salt and pepper noise
            sp_img = self.add_salt_pepper_noise(sample_images[i:i+1], 0.1)[0]
            axes[i, 2].imshow(sp_img)
            axes[i, 2].set_title('Salt & Pepper')
            axes[i, 2].axis('off')
            
            # Occlusion
            occ_img = self.add_occlusion(sample_images[i:i+1], 0.3)[0]
            axes[i, 3].imshow(occ_img)
            axes[i, 3].set_title('Occlusion')
            axes[i, 3].axis('off')
            
            # Blur
            blur_img = self.add_blur(sample_images[i:i+1], 5)[0]
            axes[i, 4].imshow(blur_img)
            axes[i, 4].set_title('Blur')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        save_path = VISUALIZATIONS_DIR / "adversarial_attacks.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attack visualizations saved to {save_path}")
    
    def plot_robustness_results(self) -> None:
        """Plot robustness testing results."""
        if not self.results:
            logger.error("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot noise robustness
        if 'noise_robustness' in self.results:
            noise_results = self.results['noise_robustness']
            
            # Gaussian noise
            gaussian_acc = list(noise_results['gaussian_noise'].values())
            gaussian_stds = [float(k.split('_')[1]) for k in noise_results['gaussian_noise'].keys()]
            axes[0, 0].plot(gaussian_stds, gaussian_acc, 'o-', label='Gaussian Noise')
            axes[0, 0].set_xlabel('Noise Standard Deviation')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Gaussian Noise Robustness')
            axes[0, 0].grid(True)
            
            # Occlusion
            occlusion_acc = list(noise_results['occlusion'].values())
            occlusion_sizes = [float(k.split('_')[1]) for k in noise_results['occlusion'].keys()]
            axes[0, 1].plot(occlusion_sizes, occlusion_acc, 'o-', label='Occlusion')
            axes[0, 1].set_xlabel('Occlusion Size (fraction)')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Occlusion Robustness')
            axes[0, 1].grid(True)
        
        # Plot adversarial attacks
        if 'adversarial_attacks' in self.results:
            adv_results = self.results['adversarial_attacks']
            
            # FGSM
            fgsm_acc = list(adv_results['fgsm'].values())
            fgsm_eps = [float(k.split('_')[1]) for k in adv_results['fgsm'].keys()]
            axes[1, 0].plot(fgsm_eps, fgsm_acc, 'o-', label='FGSM')
            axes[1, 0].set_xlabel('Epsilon')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('FGSM Attack Robustness')
            axes[1, 0].grid(True)
        
        # Plot confidence analysis
        if 'confidence_analysis' in self.results:
            conf_results = self.results['confidence_analysis']
            conditions = list(conf_results.keys())
            avg_confidences = [conf_results[c]['avg_confidence'] for c in conditions]
            
            axes[1, 1].bar(conditions, avg_confidences)
            axes[1, 1].set_xlabel('Attack Type')
            axes[1, 1].set_ylabel('Average Confidence')
            axes[1, 1].set_title('Confidence Analysis')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = VISUALIZATIONS_DIR / "robustness_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Robustness results plot saved to {save_path}")

def main():
    """Main function for testing adversarial robustness."""
    from .data_preprocessing import DataPreprocessor
    from .model_training import VehicleClassifier
    
    # Load data and model
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_generators()
    
    classifier = VehicleClassifier()
    classifier.load_model()
    
    # Get test data
    x_test, y_test = preprocessor.test_data
    
    # Initialize adversarial tester
    tester = AdversarialTester(classifier.model)
    
    # Run tests
    noise_results = tester.test_noise_robustness(x_test, y_test)
    adv_results = tester.test_adversarial_attacks(x_test, y_test)
    conf_analysis = tester.analyze_confidence_drops(x_test, y_test)
    
    # Create visualizations
    tester.visualize_attacks(x_test, y_test)
    tester.plot_robustness_results()
    
    print("Adversarial testing completed!")
    print(f"Original accuracy: {conf_analysis['original']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
