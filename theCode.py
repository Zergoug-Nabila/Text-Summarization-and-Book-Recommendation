from transformers import BartForConditionalGeneration, BartTokenizer

# Définir les modèles et tokenizers pour chaque langue
models = {
    'Englais': {
        'model_name': 'facebook/bart-large-cnn',
        'save_path': 'saved_model_en',
        'tokenizer_path': 'saved_tokenizer_en'
    },
    'Francais': {
        'model_name': 'facebook/bart-large-cnn',  # Utilisation d'un modèle préentraîné en français
        'save_path': 'saved_model_fr',
        'tokenizer_path': 'saved_tokenizer_fr'
    }
}

for lang, paths in models.items():
    model_name = paths['model_name']
    save_path = paths['save_path']
    tokenizer_path = paths['tokenizer_path']
    
    # Charger et sauvegarder le modèle et le tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(tokenizer_path)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Modèle et tokenizer pour {lang} sauvegardés.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
