import random
import re
import numpy as np
import os

# Medical acronyms dictionary
MEDICAL_ACRONYMS = {
    'hemoglobin': 'hgb',
    'hgb': 'hemoglobin',
    'white blood cell': 'wbc',
    'wbc': 'white blood cell',
    'platelet': 'plt',
    'plt': 'platelet',
    'sodium': 'na',
    'na': 'sodium',
    'potassium': 'k',
    'k': 'potassium',
    'carbon dioxide': 'co2',
    'co2': 'carbon dioxide',
    'calcium': 'ca',
    'ca': 'calcium',
    'chloride': 'cl',
    'cl': 'chloride',
    'glucose': 'gluc',
    'gluc': 'glucose',
    'blood urea nitrogen': 'bun',
    'bun': 'blood urea nitrogen',
    'creatinine': 'crea',
    'crea': 'creatinine',
    'alanine aminotransferase': 'alt',
    'alt': 'alanine aminotransferase',
    'aspartate aminotransferase': 'ast',
    'ast': 'aspartate aminotransferase',
    'alkaline phosphatase': 'alp',
    'alp': 'alkaline phosphatase',
    'total bilirubin': 'tbil',
    'tbil': 'total bilirubin',
    'direct bilirubin': 'dbil',
    'dbil': 'direct bilirubin',
    'phosphorus': 'phos',
    'phos': 'phosphorus',
    'magnesium': 'mg',
    'mg': 'magnesium',
    'albumin': 'alb',
    'alb': 'albumin',
    'protein': 'prot',
    'prot': 'protein',
    'lactate dehydrogenase': 'ldh',
    'ldh': 'lactate dehydrogenase',
    'uric acid': 'ua',
    'ua': 'uric acid',
    'white blood cell count': 'wbc',
    'leukocytes': 'wbc',
    'platelets': 'plt',
    'thrombocytes': 'plt',
    'red blood cell': 'rbc',
    'rbc': 'red blood cell',
    'serum': 'ser',
    'ser': 'serum',
    'plasma': 'plas',
    'plas': 'plasma',
    'urine': 'ur',
    'ur': 'urine',
    'blood': 'bld',
    'bld': 'blood',
    'mass/volume': 'mcnc',
    'mcnc': 'mass/volume',
    'moles/volume': 'scnc',
    'scnc': 'moles/volume',
    'presence': 'ql',
    'ql': 'presence',
}

def char_random_deletion(text, p=0.1):
    """
    Randomly remove characters from the text with probability p
    
    Args:
        text: Input text string
        p: Probability of deletion for each character
        
    Returns:
        Augmented text with some characters removed
    """
    if not text:
        return text
        
    chars = list(text)
    # Don't delete too many characters
    i = 0
    while i < len(chars):
        if random.random() < p:
            chars.pop(i)
        else:
            i += 1
            
    # Don't return empty string
    if not chars:
        return text
        
    return ''.join(chars)

def word_random_swapping(text, max_swaps=2):
    """
    Randomly swap adjacent words in the text
    
    Args:
        text: Input text string
        max_swaps: Maximum number of swaps to perform
        
    Returns:
        Augmented text with some words swapped
    """
    if not text:
        return text
        
    words = text.split()
    if len(words) <= 1:
        return text
        
    num_swaps = min(max_swaps, len(words) // 2)
    for _ in range(num_swaps):
        i = random.randint(0, len(words) - 2)
        words[i], words[i+1] = words[i+1], words[i]
        
    return ' '.join(words)

def word_random_insertion(text, related_terms=None, max_inserts=2):
    """
    Randomly insert words from related terms into the text
    
    Args:
        text: Input text string
        related_terms: List of related terms to insert
        max_inserts: Maximum number of insertions to perform
        
    Returns:
        Augmented text with some words inserted
    """
    if not text or not related_terms:
        return text
        
    words = text.split()
    
    # Parse related terms into a list
    if isinstance(related_terms, str):
        related_words = related_terms.split()
    else:
        related_words = []
        for term in related_terms:
            if term and isinstance(term, str):
                related_words.extend(term.split())
    
    if not related_words:
        return text
        
    num_inserts = min(max_inserts, len(words))
    for _ in range(num_inserts):
        insert_pos = random.randint(0, len(words))
        insert_word = random.choice(related_words)
        words.insert(insert_pos, insert_word)
        
    return ' '.join(words)

def acronym_substitution(text):
    """
    Substitute words/phrases with their acronyms or vice versa
    
    Args:
        text: Input text string
        
    Returns:
        Augmented text with acronym substitutions
    """
    if not text:
        return text
        
    # Check each key in the acronym dictionary
    result = text.lower()
    for phrase, acronym in MEDICAL_ACRONYMS.items():
        # Randomly decide whether to replace
        if random.random() < 0.5 and re.search(r'\b' + re.escape(phrase) + r'\b', result):
            result = re.sub(r'\b' + re.escape(phrase) + r'\b', acronym, result)
            
    return result

def augment_text(text_list, num_variants=5):
    """
    Apply multiple augmentation techniques to create variations of the input texts
    
    Args:
        text_list: List of text strings to augment
        num_variants: Number of augmented variants to create for each input
        
    Returns:
        List of augmented texts
    """
    if not text_list:
        return []
        
    augmented_texts = []
    
    for text in text_list:
        if not text or not isinstance(text, str):
            continue
            
        # Add the original text
        augmented_texts.append(text.lower())
        
        # Create additional variants
        for _ in range(num_variants - 1):
            # Apply a random sequence of augmentations
            augmented_text = text.lower()
            
            # Randomly apply each augmentation with 50% probability
            if random.random() < 0.5:
                augmented_text = char_random_deletion(augmented_text)
                
            if random.random() < 0.5:
                augmented_text = word_random_swapping(augmented_text)
                
            if random.random() < 0.5:
                # For word insertion, use the original text as related terms
                augmented_text = word_random_insertion(augmented_text, [text])
                
            if random.random() < 0.5:
                augmented_text = acronym_substitution(augmented_text)
                
            augmented_texts.append(augmented_text)
    
    return augmented_texts

if __name__ == "__main__":
    # Test the augmentation functions
    test_texts = [
        "hemoglobin [mass/volume] in blood",
        "white blood cell count",
        "potassium serum",
        "platelet count"
    ]
    
    print("Original texts:")
    for text in test_texts:
        print(f"  - {text}")
    
    print("\nAugmented texts:")
    augmented = augment_text(test_texts, num_variants=3)
    for text in augmented:
        print(f"  - {text}") 