# check_dataset_sizes.py
from datasets import load_dataset

dataset = load_dataset("nayeon212/BLEnD", "multiple-choice-questions")

# Count by country
countries = {}
for item in dataset['test']:
    country = item['country']
    countries[country] = countries.get(country, 0) + 1

print("\nBLEnD Dataset - Samples per Country:")
print("="*50)
for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
    print(f"{country:30s}: {count:6,} samples")
    
print(f"\nTotal: {len(dataset['test']):,} samples")