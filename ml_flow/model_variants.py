from typing import List, Dict, Any
from src.utils.config import load_config


def generate_experiment_variants() -> List[Dict[str, Any]]:
    config = load_config("model")
    model_config = config.get('model', {})
    experiments = []
    

    variants_mapping = {
        'logistic_regression': [
            {'max_iter': 500, 'class_weight': 'balanced'},
            {'max_iter': 1000, 'class_weight': None},
        ],
        'knn': [
            {'n_neighbors': 5, 'weights': 'distance'},
            {'n_neighbors': 7, 'weights': 'uniform'},
        ],
        'gradient_boosting': [
            {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 3},
            {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4},
        ],
        'random_forest': [
            {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5},
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 2},
        ],
    }
    
    for model_name, variants in variants_mapping.items():
        if model_name in model_config:
            base_config = model_config.get(model_name, {})
            
            for idx, variant_params in enumerate(variants, 1):
                full_config = {**base_config, **variant_params}
                
                experiment = {
                    "name": f"{model_name}_v{idx}",
                    model_name: full_config
                }
                experiments.append(experiment)
    
    return experiments


if __name__ == "__main__":
    variants = generate_experiment_variants()
    for exp in variants:
        print(f"\n{exp['name']}:")
        print(exp[exp['name']])
