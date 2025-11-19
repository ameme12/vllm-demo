from repos.repo_manager import RepoManager
from pathlib import Path

print("Testing with CARROT repository...")

manager = RepoManager()

# Clone CARROT
print("\n1. Cloning CARROT...")
carrot_path = manager.clone_or_update("https://github.com/TenneyHu/CARROT")
print(f"   ✓ Location: {carrot_path}")

# Check structure
print("\n2. Checking structure...")
if carrot_path.exists():
    items = list(carrot_path.iterdir())
    print(f"   ✓ Found {len(items)} items in repository")
    
    for item in items:
        if item.is_dir():
            sub_items = len(list(item.iterdir()))
            print(f"     📁 {item.name}/ ({sub_items} items)")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"     📄 {item.name} ({size_kb:.1f} KB)")
else:
    print("   ✗ CARROT not found!")
    exit(1)

# Look for data files
print("\n3. Looking for data files...")
data_dirs = ["data", "dataset", "datasets", "examples"]
found_data = False

for dirname in data_dirs:
    data_path = carrot_path / dirname
    if data_path.exists():
        print(f"   ✓ Found: {dirname}/")
        files = list(data_path.glob("*"))
        for f in files[:5]:  # Show first 5
            print(f"     - {f.name}")
        found_data = True

if not found_data:
    print("   ℹ No standard data directory found")
    print("   You may need to check the repo structure manually")

# Check for requirements.txt
print("\n4. Checking for requirements.txt...")
req_file = carrot_path / "requirements.txt"
if req_file.exists():
    print(f"   ✓ Found requirements.txt")
    with open(req_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    print(f"   Dependencies: {len(lines)}")
    for line in lines[:5]:
        print(f"     - {line}")
else:
    print("   ℹ No requirements.txt")

print("\n✅ CARROT test complete!")
print(f"\nCARROT is ready at: {carrot_path}")
print("You can now use this path in your tasks!")