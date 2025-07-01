from flask import Flask, render_template, request, jsonify, render_template_string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
from deap import base, creator, tools, algorithms
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import warnings
import io
import base64
import json
import pickle
import os
from datetime import datetime
from werkzeug.utils import secure_filename
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
PRETRAINED_MODEL_PATH = 'pretrained_model.pkl'
DATA_SNAPSHOT_PATH = 'data_snapshot.csv'
UPLOAD_FOLDER = 'uploaded_data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

uploaded_dataset_path = None

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class SymbolicRule:
    """Represents a symbolic IF-THEN rule for medical diagnosis"""
    feature: str
    operator: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    prediction: int
    confidence: float
    
    def evaluate(self, patient_data: Dict[str, float]) -> Tuple[bool, int, float]:
        """Evaluate rule against patient data"""
        feature_value = patient_data.get(self.feature, 0)
        
        if self.operator == '>':
            applies = feature_value > self.threshold
        elif self.operator == '<':
            applies = feature_value < self.threshold
        elif self.operator == '>=':
            applies = feature_value >= self.threshold
        elif self.operator == '<=':
            applies = feature_value <= self.threshold
        else:  # ==
            applies = abs(feature_value - self.threshold) < 0.1
            
        return applies, self.prediction, self.confidence
    
    def __str__(self):
        return f"IF {self.feature} {self.operator} {self.threshold:.2f} THEN diagnosis={self.prediction} (conf={self.confidence:.2f})"

class MedicalDatasetGenerator:
    """Generate synthetic medical dataset with realistic patterns"""
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.feature_names = [
            'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'cholesterol', 'blood_sugar', 'heart_rate', 'bmi',
            'exercise_hours_per_week', 'smoking_years', 'family_history'
        ]
        self.conditions = ['Healthy', 'Diabetes', 'Heart Disease', 'Hypertension']
        
    def generate_dataset(self):
        """Generate realistic medical dataset"""
        data = []
        labels = []
        
        for _ in range(self.n_samples):
            # Base patient profile
            age = np.random.normal(45, 15)
            age = max(18, min(90, age))
            
            # Generate correlated features
            # Healthy patients (25%)
            if np.random.random() < 0.25:
                bp_sys = np.random.normal(120, 10)
                bp_dia = np.random.normal(80, 8)
                cholesterol = np.random.normal(180, 20)
                blood_sugar = np.random.normal(90, 10)
                heart_rate = np.random.normal(70, 10)
                bmi = np.random.normal(22, 3)
                exercise = np.random.normal(4, 2)
                smoking = 0 if np.random.random() > 0.2 else np.random.normal(2, 3)
                family_hist = 1 if np.random.random() < 0.3 else 0
                label = 0  # Healthy
                
            # Diabetes patients (25%)
            elif np.random.random() < 0.33:
                bp_sys = np.random.normal(140, 15)
                bp_dia = np.random.normal(90, 10)
                cholesterol = np.random.normal(220, 30)
                blood_sugar = np.random.normal(160, 40)  # High blood sugar
                heart_rate = np.random.normal(75, 12)
                bmi = np.random.normal(28, 4)  # Higher BMI
                exercise = np.random.normal(2, 1.5)  # Less exercise
                smoking = np.random.normal(8, 5) if np.random.random() > 0.4 else 0
                family_hist = 1 if np.random.random() < 0.6 else 0  # Higher family history
                label = 1  # Diabetes
                
            # Heart Disease patients (25%)
            elif np.random.random() < 0.5:
                bp_sys = np.random.normal(150, 20)
                bp_dia = np.random.normal(95, 12)
                cholesterol = np.random.normal(240, 35)  # High cholesterol
                blood_sugar = np.random.normal(100, 15)
                heart_rate = np.random.normal(80, 15)
                bmi = np.random.normal(27, 4)
                exercise = np.random.normal(1.5, 1)  # Low exercise
                smoking = np.random.normal(12, 6) if np.random.random() > 0.3 else 0
                family_hist = 1 if np.random.random() < 0.7 else 0
                label = 2  # Heart Disease
                
            # Hypertension patients (25%)
            else:
                bp_sys = np.random.normal(160, 20)  # High BP
                bp_dia = np.random.normal(100, 15)  # High BP
                cholesterol = np.random.normal(200, 25)
                blood_sugar = np.random.normal(95, 12)
                heart_rate = np.random.normal(75, 12)
                bmi = np.random.normal(26, 4)
                exercise = np.random.normal(2.5, 1.5)
                smoking = np.random.normal(6, 4) if np.random.random() > 0.5 else 0
                family_hist = 1 if np.random.random() < 0.5 else 0
                label = 3  # Hypertension
            
            # Ensure realistic bounds
            patient = [
                max(18, min(90, age)),
                max(80, min(200, bp_sys)),
                max(50, min(120, bp_dia)),
                max(120, min(350, cholesterol)),
                max(60, min(300, blood_sugar)),
                max(50, min(120, heart_rate)),
                max(15, min(45, bmi)),
                max(0, min(10, exercise)),
                max(0, smoking),  # Can be 0
                family_hist
            ]
            
            data.append(patient)
            labels.append(label)
        
        df = pd.DataFrame(data, columns=self.feature_names)
        df['diagnosis'] = labels
        return df

class NeuroSymbolicEvolutionarySystem:
    """Main system combining neural networks with evolutionary symbolic rules"""
    
    def __init__(self):
        self.neural_model = None
        self.symbolic_rules = []
        self.feature_names = []
        self.scaler = StandardScaler()
        self.conditions = ['Healthy', 'Diabetes', 'Heart Disease', 'Hypertension']
        self.is_trained = False
        self.training_date = None
        self.accuracy = 0.0
        
        # Setup DEAP for evolutionary algorithm
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
    
    def save_model(self, file_path):
        """Save the trained model to a file"""
        model_data = {
            'neural_model': self.neural_model,
            'symbolic_rules': self.symbolic_rules,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'accuracy': self.accuracy
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, file_path):
        """Load a trained model from file"""
        try:
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.neural_model = model_data['neural_model']
            self.symbolic_rules = model_data['symbolic_rules']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.training_date = model_data['training_date']
            self.accuracy = model_data['accuracy']
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def prepare_data(self, df):
        """Prepare dataset for training"""
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_neural_model(self, X_train, y_train):
        """Train neural network component"""
        print("Training Neural Network...")
        self.neural_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.neural_model.fit(X_train, y_train)
        print(f"Neural Network Training Complete. Iterations: {self.neural_model.n_iter_}")
    
    def create_individual(self):
        """Create a random symbolic rule (individual for evolution)"""
        feature = random.choice(self.feature_names)
        operator = random.choice(['>', '<', '>=', '<='])
        
        # Set reasonable thresholds based on feature
        if feature == 'age':
            threshold = random.uniform(20, 80)
        elif 'blood_pressure' in feature:
            threshold = random.uniform(80, 180)
        elif feature == 'cholesterol':
            threshold = random.uniform(150, 300)
        elif feature == 'blood_sugar':
            threshold = random.uniform(70, 200)
        elif feature == 'heart_rate':
            threshold = random.uniform(50, 120)
        elif feature == 'bmi':
            threshold = random.uniform(18, 40)
        elif feature == 'exercise_hours_per_week':
            threshold = random.uniform(0, 8)
        elif feature == 'smoking_years':
            threshold = random.uniform(0, 30)
        else:  # family_history
            threshold = 0.5
            
        prediction = random.randint(0, 3)
        confidence = random.uniform(0.5, 1.0)
        
        return [feature, operator, threshold, prediction, confidence]
    
    def evaluate_rule(self, individual, X_data, y_true):
        """Evaluate fitness of a symbolic rule"""
        feature, operator, threshold, prediction, confidence = individual
        
        correct_predictions = 0
        total_applicable = 0
        
        for idx, (i, row) in enumerate(X_data.iterrows()):
            patient_data = row.to_dict()
            rule = SymbolicRule(feature, operator, threshold, prediction, confidence)
            applies, pred, conf = rule.evaluate(patient_data)
            
            if applies:
                total_applicable += 1
                if pred == y_true.iloc[idx]:
                    correct_predictions += 1
        
        if total_applicable == 0:
            return (0.0,)  # Rule doesn't apply to any cases
        
        accuracy = correct_predictions / total_applicable
        coverage = total_applicable / len(X_data)
        
        # Fitness combines accuracy and coverage
        fitness = accuracy * 0.7 + coverage * 0.3
        return (fitness,)
    
    def evolve_symbolic_rules(self, X_train_raw, y_train, generations=15):
        """Evolve symbolic rules using genetic algorithm"""
        print("Evolving Symbolic Rules...")
        
        # Setup toolbox
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_rule, X_data=X_train_raw, y_true=y_train)
        self.toolbox.register("mate", self.crossover_rules)
        self.toolbox.register("mutate", self.mutate_rule)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = self.toolbox.population(n=50)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for gen in range(generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            population[:] = offspring
        
        # Extract best rules
        best_individuals = tools.selBest(population, k=10)
        self.symbolic_rules = []
        
        for ind in best_individuals:
            feature, operator, threshold, prediction, confidence = ind
            rule = SymbolicRule(feature, operator, threshold, prediction, confidence)
            self.symbolic_rules.append(rule)
        
        print(f"Evolution Complete. Generated {len(self.symbolic_rules)} rules.")
    
    def crossover_rules(self, ind1, ind2):
        """Crossover operation for symbolic rules"""
        # Swap random components
        if random.random() < 0.5:
            ind1[0], ind2[0] = ind2[0], ind1[0]  # feature
        if random.random() < 0.5:
            ind1[1], ind2[1] = ind2[1], ind1[1]  # operator
        if random.random() < 0.5:
            ind1[2], ind2[2] = ind2[2], ind1[2]  # threshold
        if random.random() < 0.5:
            ind1[3], ind2[3] = ind2[3], ind1[3]  # prediction
        
        return ind1, ind2
    
    def mutate_rule(self, individual):
        """Mutation operation for symbolic rules"""
        mutation_type = random.randint(0, 4)
        
        if mutation_type == 0:  # Mutate feature
            individual[0] = random.choice(self.feature_names)
        elif mutation_type == 1:  # Mutate operator
            individual[1] = random.choice(['>', '<', '>=', '<='])
        elif mutation_type == 2:  # Mutate threshold
            individual[2] += random.gauss(0, individual[2] * 0.1)
        elif mutation_type == 3:  # Mutate prediction
            individual[3] = random.randint(0, 3)
        else:  # Mutate confidence
            individual[4] = max(0.1, min(1.0, individual[4] + random.gauss(0, 0.1)))
        
        return (individual,)
    
    def hybrid_predict(self, X_test_scaled, X_test_raw):
        """Make predictions using hybrid neuro-symbolic approach"""
        neural_preds = self.neural_model.predict_proba(X_test_scaled)
        hybrid_preds = []
        explanations = []
        
        for i, (neural_prob, row) in enumerate(zip(neural_preds, X_test_raw.iterrows())):
            patient_data = row[1].to_dict()
            
            # Check if any symbolic rule applies
            applicable_rules = []
            for rule in self.symbolic_rules:
                applies, pred, conf = rule.evaluate(patient_data)
                if applies:
                    applicable_rules.append((rule, pred, conf))
            
            if applicable_rules:
                # Use symbolic rule with highest confidence
                best_rule, symbolic_pred, confidence = max(applicable_rules, key=lambda x: x[2])
                
                # Combine neural and symbolic predictions
                neural_pred = np.argmax(neural_prob)
                neural_confidence = np.max(neural_prob)
                
                # Weighted combination
                if confidence > neural_confidence:
                    final_pred = symbolic_pred
                    explanation = f"Symbolic Rule: {best_rule}"
                else:
                    final_pred = neural_pred
                    explanation = f"Neural Network (conf: {neural_confidence:.3f})"
            else:
                # Use neural network prediction
                final_pred = np.argmax(neural_prob)
                explanation = f"Neural Network (conf: {np.max(neural_prob):.3f})"
            
            hybrid_preds.append(final_pred)
            explanations.append(explanation)
        
        return np.array(hybrid_preds), explanations
    
    def predict_single_patient(self, patient_data):
        """Make prediction for a single patient"""
        if not self.is_trained:
            return None, "System not trained yet", None
        
        # Convert to DataFrame for scaling
        df = pd.DataFrame([patient_data])
        
        # Ensure columns are in the same order as during training
        df = df[self.feature_names]
        
        df_scaled = self.scaler.transform(df)
        
        # Get neural network prediction
        neural_prob = self.neural_model.predict_proba(df_scaled)[0]
        
        # Check symbolic rules
        applicable_rules = []
        for rule in self.symbolic_rules:
            applies, pred, conf = rule.evaluate(patient_data)
            if applies:
                applicable_rules.append((rule, pred, conf))
        
        if applicable_rules:
            # Use symbolic rule with highest confidence
            best_rule, symbolic_pred, confidence = max(applicable_rules, key=lambda x: x[2])
            
            # Combine neural and symbolic predictions
            neural_pred = np.argmax(neural_prob)
            neural_confidence = np.max(neural_prob)
            
            # Weighted combination
            if confidence > neural_confidence:
                final_pred = symbolic_pred
                explanation = f"Symbolic Rule: {best_rule}"
                recommendations = self.get_recommendations(symbolic_pred, patient_data)
            else:
                final_pred = neural_pred
                explanation = f"Neural Network (conf: {neural_confidence:.3f})"
                recommendations = self.get_recommendations(neural_pred, patient_data)
        else:
            # Use neural network prediction
            final_pred = np.argmax(neural_prob)
            explanation = f"Neural Network (conf: {np.max(neural_prob):.3f})"
            recommendations = self.get_recommendations(final_pred, patient_data)
        
        return final_pred, explanation, recommendations
    
    def get_recommendations(self, diagnosis_code, patient_data):
        """Generate personalized recommendations based on diagnosis"""
        condition = self.conditions[diagnosis_code]
        
        if condition == 'Healthy':
            return "Recommendations: Maintain your healthy lifestyle with regular exercise and balanced diet."
        elif condition == 'Diabetes':
            return f"Recommendations: Monitor blood sugar regularly. Consider reducing carbohydrate intake. Target blood sugar: {max(70, min(130, patient_data['blood_sugar'] * 0.8))} mg/dL."
        elif condition == 'Heart Disease':
            return "Recommendations: Consult a cardiologist. Reduce saturated fats and sodium. Aim for 150 mins of moderate exercise weekly."
        else:  # Hypertension
            target_bp = f"{max(110, patient_data['blood_pressure_systolic'] * 0.9)}/{max(70, patient_data['blood_pressure_diastolic'] * 0.9)}"
            return f"Recommendations: Reduce sodium intake. Monitor blood pressure daily. Target BP: {target_bp} mmHg."

# Global system instance
system = NeuroSymbolicEvolutionarySystem()

# Try to load pre-trained model if available
if os.path.exists(PRETRAINED_MODEL_PATH):
    system.load_model(PRETRAINED_MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_model_status')
def check_model_status():
    """Check if model is trained and return status"""
    return jsonify({
        'is_trained': system.is_trained,
        'accuracy': round(system.accuracy * 100, 2) if system.accuracy else None,
        'training_date': system.training_date.strftime('%Y-%m-%d %H:%M:%S') if system.training_date else None
    })

@app.route('/train', methods=['POST'])
def train():
    """Train the neuro-symbolic system"""
    try:
        # Generate synthetic dataset
        generator = MedicalDatasetGenerator(n_samples=1000)
        df = generator.generate_dataset()
        
        # Prepare data
        X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = system.prepare_data(df)
        
        # Train neural network
        system.train_neural_model(X_train_scaled, y_train)
        
        # Evolve symbolic rules
        system.evolve_symbolic_rules(X_train_raw, y_train)
        
        # Evaluate system
        hybrid_preds, _ = system.hybrid_predict(X_test_scaled, X_test_raw)
        accuracy = accuracy_score(y_test, hybrid_preds)
        system.accuracy = accuracy
        system.is_trained = True
        system.training_date = datetime.now()
        
        # Save model
        system.save_model(PRETRAINED_MODEL_PATH)
        
        # Save data snapshot
        df.to_csv(DATA_SNAPSHOT_PATH, index=False)
        
        # Format rules for display
        rules_display = [str(rule) for rule in system.symbolic_rules]
        
        return jsonify({
            'success': True,
            'message': 'System trained successfully!',
            'accuracy': round(accuracy * 100, 2),
            'training_date': system.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            'rules': rules_display
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/load_pretrained', methods=['POST'])
def load_pretrained():
    """Load pre-trained model"""
    try:
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            return jsonify({
                'success': False,
                'message': 'No pre-trained model found'
            }), 404
            
        success = system.load_model(PRETRAINED_MODEL_PATH)
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to load pre-trained model'
            }), 500
            
        # Format rules for display
        rules_display = [str(rule) for rule in system.symbolic_rules]
        
        return jsonify({
            'success': True,
            'message': 'Pre-trained model loaded successfully!',
            'accuracy': round(system.accuracy * 100, 2),
            'training_date': system.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            'rules': rules_display
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not system.is_trained:
            return jsonify({
                'success': False,
                'message': 'System is not trained yet'
            }), 400

        # Get patient data from request
        patient_data = request.json

        # Validate required fields
        required_fields = [
            'age', 'bmi', 'family_history', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'heart_rate', 'blood_sugar',
            'cholesterol', 'exercise_hours_per_week', 'smoking_years'
        ]
        for field in required_fields:
            if field not in patient_data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                }), 400

        # Convert all values to float (or int for family_history)
        for field in patient_data:
            if field == 'family_history' or field == 'smoking_years':
                patient_data[field] = int(patient_data[field])
            else:
                patient_data[field] = float(patient_data[field])

        # Make prediction
        diagnosis_code, explanation, recommendations = system.predict_single_patient(patient_data)

        if diagnosis_code is None:
            return jsonify({
                'success': False,
                'message': explanation
            }), 400

        # Format response
        diagnosis = system.conditions[diagnosis_code]
        response_html = f'''<div class="prediction-result">
    <div class="diagnosis">Diagnosis: {diagnosis}</div>
    <div class="explanation">{explanation}</div>
    <div class="recommendations">{recommendations}</div>
</div>'''

        return jsonify({
            'success': True,
            'explanation': response_html
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global uploaded_dataset_path
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploaded_dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_dataset_path)
        return jsonify({'success': True, 'message': 'File uploaded successfully'})
    else:
        return jsonify({'success': False, 'message': 'Invalid file type'}), 400

@app.route('/train_on_uploaded', methods=['POST'])
def train_on_uploaded():
    global uploaded_dataset_path
    if not uploaded_dataset_path or not os.path.exists(uploaded_dataset_path):
        return jsonify({'success': False, 'message': 'No uploaded dataset found'}), 400
    try:
        df = pd.read_csv(uploaded_dataset_path)
        # Check if required columns exist
        required_cols = [
            'age', 'bmi', 'family_history', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'heart_rate', 'blood_sugar',
            'cholesterol', 'exercise_hours_per_week', 'smoking_years', 'diagnosis'
        ]
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'success': False, 'message': f'Missing column: {col}'}), 400
        X_train_scaled, X_test_scaled, y_train, y_test, X_train_raw, X_test_raw = system.prepare_data(df)
        system.train_neural_model(X_train_scaled, y_train)
        system.evolve_symbolic_rules(X_train_raw, y_train)
        hybrid_preds, _ = system.hybrid_predict(X_test_scaled, X_test_raw)
        accuracy = accuracy_score(y_test, hybrid_preds)
        system.accuracy = accuracy
        system.is_trained = True
        system.training_date = datetime.now()
        # Save model
        system.save_model(PRETRAINED_MODEL_PATH)
        # Save data snapshot
        df.to_csv(DATA_SNAPSHOT_PATH, index=False)
        rules_display = [str(rule) for rule in system.symbolic_rules]
        return jsonify({
            'success': True,
            'message': 'Model trained on uploaded data!',
            'accuracy': round(accuracy * 100, 2),
            'training_date': system.training_date.strftime('%Y-%m-%d %H:%M:%S'),
            'rules': rules_display
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Helper to check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
