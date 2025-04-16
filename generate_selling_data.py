import random
import faker

# Initialize Faker
fake = faker.Faker()

# Define sample products and prices
products = [
    ("Laptop", 1000),
    ("Smartphone", 700),
    ("Tablet", 400),
    ("Headphones", 150),
    ("Smartwatch", 250),
    ("Camera", 850),
    ("Monitor", 300),
    ("Keyboard", 100),
    ("Mouse", 50),
    ("Printer", 200)
]

# Generate 100 rows of fake selling data
selling_data = []
for _ in range(100):
    product, base_price = random.choice(products)
    quantity = random.randint(1, 5)
    price = base_price * quantity
    date = fake.date_between(start_date='-1y', end_date='today')
    customer = fake.name()
    selling_data.append(f"{date} | {customer} | {product} | Qty: {quantity} | Total: ${price}")

# Save to .txt file
file_path = "./selling_data.txt"
with open(file_path, "w") as file:
    file.write("\n".join(selling_data))

file_path
