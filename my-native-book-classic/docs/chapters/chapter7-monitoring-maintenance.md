---
sidebar_position: 7
---

# Chapter 7: Monitoring, Maintenance, and Continuous Improvement

## AI Model Monitoring

Monitoring AI-native applications goes beyond traditional system monitoring to include model-specific metrics and behaviors. This requires tracking both the infrastructure and the model's performance in production.

### Model Performance Monitoring

#### Accuracy Tracking

Model accuracy can degrade over time due to changes in input data distribution (data drift) or concept drift. Implement continuous evaluation:

```python
import numpy as np
from datetime import datetime
from typing import Dict, List

class ModelPerformanceTracker:
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_model(self, y_true: List, y_pred: List) -> Dict:
        """Calculate key performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_count': len(y_true)
        }
        
        self.metrics_history.append(metrics)
        return metrics

# Example usage in a monitoring pipeline
tracker = ModelPerformanceTracker()

# In production: Collect ground truth when available
# y_true = get_ground_truth_from_feedback_system()
# y_pred = model_predictions
# current_metrics = tracker.evaluate_model(y_true, y_pred)
```

#### Data Quality Monitoring

Track changes in data distribution that might affect model performance:

```python
import pandas as pd
from scipy import stats
import numpy as np

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        self.reference_data = reference_data
        self.threshold = threshold
        self.features = list(reference_data.columns)
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect data drift using statistical tests."""
        drift_results = {}
        
        for feature in self.features:
            if current_data[feature].dtype in ['object', 'category']:
                # Categorical feature: use chi-square test
                ref_counts = self.reference_data[feature].value_counts(normalize=True)
                curr_counts = current_data[feature].value_counts(normalize=True)
                
                # Align indices and calculate drift
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_probs = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_probs = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Calculate JS divergence as a measure of drift
                js_divergence = self._jensen_shannon_divergence(ref_probs, curr_probs)
                drift_results[feature] = {
                    'type': 'categorical',
                    'drift_score': js_divergence,
                    'is_drifted': js_divergence > self.threshold
                }
            else:
                # Numerical feature: use KS test
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[feature], 
                    current_data[feature]
                )
                
                drift_results[feature] = {
                    'type': 'numerical',
                    'drift_score': ks_statistic,
                    'is_drifted': ks_statistic > self.threshold,
                    'p_value': p_value
                }
        
        return drift_results
    
    def _jensen_shannon_divergence(self, p, q):
        """Compute Jensen-Shannon divergence between two probability distributions."""
        m = 0.5 * (np.array(p) + np.array(q))
        return 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

# Example usage
# drift_detector = DataDriftDetector(training_data, threshold=0.05)
# current_batch_data = get_current_batch_data()
# drift_results = drift_detector.detect_drift(current_batch_data)
```

### Prediction Monitoring

#### Prediction Distribution Tracking

Monitor changes in model prediction patterns:

```python
from collections import Counter
import matplotlib.pyplot as plt
import json

class PredictionDistributionMonitor:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.prediction_history = []
    
    def record_predictions(self, predictions: List[str]):
        """Record and analyze prediction distribution."""
        counter = Counter(predictions)
        distribution = {label: counter.get(label, 0) for label in self.labels}
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'distribution': distribution,
            'total_predictions': len(predictions)
        }
        
        self.prediction_history.append(record)
        
        # Check for significant changes in prediction patterns
        if len(self.prediction_history) > 1:
            self._check_distribution_shift(record)
    
    def _check_distribution_shift(self, current_record):
        """Check for significant shifts in prediction distribution."""
        if len(self.prediction_history) < 2:
            return
        
        prev_record = self.prediction_history[-2]
        prev_dist = prev_record['distribution']
        curr_dist = current_record['distribution']
        
        # Calculate distribution difference
        total_prev = prev_record['total_predictions']
        total_curr = current_record['total_predictions']
        
        if total_prev > 0 and total_curr > 0:
            prev_ratios = {k: v/total_prev for k, v in prev_dist.items()}
            curr_ratios = {k: v/total_curr for k, v in curr_dist.items()}
            
            # Calculate maximum absolute difference in ratios
            max_diff = max(abs(prev_ratios[k] - curr_ratios[k]) for k in self.labels)
            
            if max_diff > 0.1:  # Alert if difference is > 10%
                print(f"ALERT: Significant prediction distribution shift detected: {max_diff:.3f}")
```

## Alerting and Anomaly Detection

### Automated Alerting System

Set up automated alerts for when model performance degrades or anomalies occur:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ModelAlertSystem:
    def __init__(self, smtp_server: str, sender_email: str, recipient_emails: List[str]):
        self.smtp_server = smtp_server
        self.sender_email = sender_email
        self.recipient_emails = recipient_emails
    
    def send_alert(self, subject: str, message: str):
        """Send an alert email."""
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(self.recipient_emails)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        try:
            server = smtplib.SMTP(self.smtp_server)
            server.send_message(msg)
            server.quit()
            print("Alert email sent successfully")
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
    
    def check_model_performance(self, metrics: Dict):
        """Check model metrics and trigger alerts if needed."""
        if metrics['accuracy'] < 0.8:
            self.send_alert(
                "CRITICAL: Model Accuracy Below Threshold",
                f"Current accuracy: {metrics['accuracy']:.3f}. Expected > 0.8."
            )
        
        if metrics['f1_score'] < 0.75:
            self.send_alert(
                "WARNING: Model F1 Score Degrading",
                f"Current F1 score: {metrics['f1_score']:.3f}. Investigate potential issues."
            )

# Example usage
# alert_system = ModelAlertSystem(
#     smtp_server='smtp.gmail.com',
#     sender_email='alerts@yourcompany.com',
#     recipient_emails=['ml-team@yourcompany.com', 'ops@yourcompany.com']
# )
```

### Real-time Monitoring Dashboard

While not implementing the UI here, you would typically create dashboards using:

- **Grafana**: For metric visualization
- **Kibana**: For log analysis
- **Custom Dashboards**: Using React/Python frameworks
- **Prometheus**: For metrics collection

## Model Maintenance and Retraining

### Automated Retraining Pipelines

Implement automated retraining when performance degrades:

```python
from datetime import datetime, timedelta
import subprocess

class AutomatedRetrainingSystem:
    def __init__(self, model_path: str, training_script: str, performance_threshold: float = 0.85):
        self.model_path = model_path
        self.training_script = training_script
        self.performance_threshold = performance_threshold
        self.last_training_time = datetime.now()
        self.retraining_cooldown = timedelta(hours=24)  # Don't retrain too frequently
    
    def should_retrain(self, current_metrics: Dict) -> bool:
        """Determine if model retraining is needed."""
        if datetime.now() - self.last_training_time < self.retraining_cooldown:
            # Still in cooldown period
            return False
        
        if current_metrics['accuracy'] < self.performance_threshold:
            return True
        
        # Check if there's been significant data drift
        # This would be connected to the drift detection system
        if hasattr(self, 'has_significant_drift') and self.has_significant_drift:
            return True
        
        return False
    
    def trigger_retraining(self, new_data_path: str):
        """Trigger the retraining process."""
        print("Starting automated retraining...")
        
        # Run the training script
        cmd = [
            'python', 
            self.training_script,
            '--data', new_data_path,
            '--output', self.model_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Retraining completed successfully")
            self.last_training_time = datetime.now()
            
            # Validate the new model before deploying
            if self._validate_new_model():
                self._deploy_model()
            else:
                print("New model validation failed, keeping old model")
        else:
            print(f"Retraining failed: {result.stderr}")
    
    def _validate_new_model(self) -> bool:
        """Validate the new model before deployment."""
        # Load new model and test on validation set
        # Return True if it meets quality criteria
        pass
    
    def _deploy_model(self):
        """Deploy the new model to production."""
        # Implementation would depend on model serving infrastructure
        # e.g., update model files, restart services, etc.
        pass
```

### Model Versioning and Rollback

```python
import shutil
import os
from datetime import datetime

class ModelVersionManager:
    def __init__(self, model_directory: str):
        self.model_directory = model_directory
        self.versions_directory = os.path.join(model_directory, 'versions')
        
        # Ensure versions directory exists
        os.makedirs(self.versions_directory, exist_ok=True)
    
    def save_model_version(self, model_path: str, version_description: str = ""):
        """Save a new version of the model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"model_{timestamp}"
        version_path = os.path.join(self.versions_directory, version_name)
        
        # Create version directory
        os.makedirs(version_path, exist_ok=True)
        
        # Copy model files to version directory
        shutil.copy2(model_path, version_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'version': version_name,
            'description': version_description,
            'original_path': model_path
        }
        
        with open(os.path.join(version_path, 'metadata.json'), 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"Model version saved: {version_name}")
        return version_name
    
    def list_versions(self):
        """List all available model versions."""
        versions = os.listdir(self.versions_directory)
        return sorted(versions)
    
    def rollback_to_version(self, version_name: str, target_path: str):
        """Rollback to a specific model version."""
        version_path = os.path.join(self.versions_directory, version_name)
        model_files = [f for f in os.listdir(version_path) if f != 'metadata.json']
        
        if not model_files:
            raise ValueError(f"No model files found in version {version_name}")
        
        # Copy model file back to target path
        source_model = os.path.join(version_path, model_files[0])
        shutil.copy2(source_model, target_path)
        
        print(f"Rolled back to version: {version_name}")
```

## Continuous Improvement

### Feedback Collection and Integration

Implement systems to collect and use feedback for model improvement:

```python
class FeedbackCollector:
    def __init__(self, feedback_storage_path: str):
        self.feedback_storage_path = feedback_storage_path
        self.feedback_buffer = []
        self.buffer_size = 1000
    
    def collect_feedback(self, prediction_id: str, user_input: str, is_correct: bool, explanation: str = ""):
        """Collect user feedback on model predictions."""
        feedback_record = {
            'prediction_id': prediction_id,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'is_correct': is_correct,
            'explanation': explanation
        }
        
        self.feedback_buffer.append(feedback_record)
        
        # If buffer is full, save to storage
        if len(self.feedback_buffer) >= self.buffer_size:
            self._save_feedback()
    
    def _save_feedback(self):
        """Save feedback to persistent storage."""
        import json
        os.makedirs(os.path.dirname(self.feedback_storage_path), exist_ok=True)
        
        with open(self.feedback_storage_path, 'a') as f:
            for record in self.feedback_buffer:
                f.write(json.dumps(record) + '\n')
        
        self.feedback_buffer = []
        print(f"Saved {len(self.feedback_buffer)} feedback records")
    
    def get_feedback_for_retraining(self, min_feedback_count: int = 100):
        """Retrieve feedback for model improvement."""
        # Implementation would read feedback from storage
        # and potentially filter based on various criteria
        pass

# Example of integrating feedback into model improvement
class FeedbackBasedImprovement:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.feedback_collector = FeedbackCollector('feedback/feedback.jsonl')
    
    def analyze_feedback_patterns(self):
        """Analyze feedback to identify model weaknesses."""
        # Implement analysis logic to find common failure patterns
        # This could reveal data that needs to be collected
        # or features that need to be added
        pass
    
    def generate_training_examples(self, feedback_data):
        """Generate new training examples from feedback."""
        # Convert feedback into training data format
        # This is particularly useful for supervised learning models
        pass
```

### Model Lifecycle Management

Implement a complete model lifecycle management system:

```python
from enum import Enum
from typing import Optional

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

class ModelLifecycleManager:
    def __init__(self):
        self.models = {}
    
    def register_model(self, model_id: str, model_path: str, version: str, performance_metrics: Dict):
        """Register a new model version."""
        self.models[model_id] = {
            'id': model_id,
            'path': model_path,
            'version': version,
            'status': ModelStatus.DEVELOPMENT,
            'performance_metrics': performance_metrics,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def promote_model(self, model_id: str, new_status: ModelStatus):
        """Promote model to next lifecycle stage."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if self._can_promote(model_id, new_status):
            self.models[model_id]['status'] = new_status
            self.models[model_id]['last_updated'] = datetime.now().isoformat()
            print(f"Model {model_id} promoted to {new_status.value}")
        else:
            raise ValueError(f"Cannot promote model {model_id} to {new_status.value}")
    
    def _can_promote(self, model_id: str, new_status: ModelStatus) -> bool:
        """Check if model can be promoted to new status."""
        current_status = self.models[model_id]['status']
        
        # Define promotion rules
        if current_status == ModelStatus.DEVELOPMENT and new_status == ModelStatus.STAGING:
            # Check if model meets staging criteria
            accuracy = self.models[model_id]['performance_metrics'].get('accuracy', 0)
            return accuracy >= 0.85
        elif current_status == ModelStatus.STAGING and new_status == ModelStatus.PRODUCTION:
            # Check if model meets production criteria
            accuracy = self.models[model_id]['performance_metrics'].get('accuracy', 0)
            stability = self.models[model_id]['performance_metrics'].get('stability_score', 0)
            return accuracy >= 0.90 and stability >= 0.95
        elif current_status == ModelStatus.PRODUCTION and new_status == ModelStatus.DEPRECATED:
            # Any production model can be deprecated
            return True
        
        return False
    
    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model."""
        for model in self.models.values():
            if model['status'] == ModelStatus.PRODUCTION:
                return model
        return None
```

## Performance Optimization

### A/B Testing Framework for Models

Implement A/B testing to compare different model versions:

```python
import random
from typing import Any

class ModelABTestFramework:
    def __init__(self):
        self.models = {}
        self.weights = {}  # Traffic distribution weights
    
    def register_model(self, model_id: str, model_obj: Any, weight: float = 1.0):
        """Register a model for A/B testing."""
        self.models[model_id] = model_obj
        self.weights[model_id] = weight
    
    def route_request(self, input_data: Any) -> tuple:
        """Route request to appropriate model based on weights."""
        total_weight = sum(self.weights.values())
        rand = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for model_id, weight in self.weights.items():
            cumulative_weight += weight
            if rand <= cumulative_weight:
                selected_model = self.models[model_id]
                prediction = selected_model.predict(input_data)
                return prediction, model_id
        
        # Fallback (should not happen if weights sum correctly)
        model_id = list(self.models.keys())[0]
        prediction = self.models[model_id].predict(input_data)
        return prediction, model_id
    
    def evaluate_models(self, test_data: List) -> Dict:
        """Evaluate all models with test data."""
        results = {}
        
        for model_id, model in self.models.items():
            predictions = [model.predict(x) for x in test_data]
            # Calculate metrics
            accuracy = calculate_accuracy(predictions, [y for x, y in test_data])
            latency = measure_latency(model, test_data)
            
            results[model_id] = {
                'accuracy': accuracy,
                'latency': latency
            }
        
        return results

# Example usage
# ab_test = ModelABTestFramework()
# ab_test.register_model('v1-model', old_model, weight=0.9)  # 90% traffic
# ab_test.register_model('v2-model', new_model, weight=0.1)  # 10% traffic
```

The next chapter will focus on security, privacy, and ethical considerations in AI-native applications.