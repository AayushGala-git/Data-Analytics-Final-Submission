import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import pandas as pd
import os

# Activate the pandas conversion
pandas2ri.activate()

# Set a CRAN mirror first to avoid the error
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)  # Using the first mirror in the list

# Install & load required packages
utils.install_packages('pscl')
pscl = rpackages.importr('pscl')

print("Loading biochemists data:")
# Load the biochemists data
bio = pscl.__rdata__.fetch('bioChemists')['bioChemists']
# Fix: Use the correct conversion method based on your rpy2 version
bio_df = pandas2ri.rpy2py(bio)
print(bio_df.head())

# Save the biochemists data to CSV
bio_csv_path = os.path.join(os.getcwd(), 'biochemists_data.csv')
bio_df.to_csv(bio_csv_path, index=False)
print(f"Biochemists data saved to: {bio_csv_path}")

# Now let's load some Irish data
# There's no specific "Ireland" dataset in base R packages
# Let's install a package with geographical data that might include Ireland
print("\nLoading Irish data:")
utils.install_packages('MASS')
mass = rpackages.importr('MASS')

# Check for datasets that might have Irish data or create some example data
robjects.r('''
# Create a sample dataset for Ireland
ireland_data <- data.frame(
  county = c("Dublin", "Cork", "Galway", "Limerick", "Waterford"),
  population = c(1347359, 542868, 258058, 194899, 116176),
  area_sqkm = c(921, 7500, 6149, 2756, 1857),
  gdp_million_euro = c(84, 24, 8, 7, 5)
)
''')

# Get the data from R to Python
ireland_data = robjects.r('ireland_data')
# Fix: Use the correct conversion method based on your rpy2 version
ireland_df = pandas2ri.rpy2py(ireland_data)
print(ireland_df.head())

# Save the Ireland data to CSV
ireland_csv_path = os.path.join(os.getcwd(), 'ireland_data.csv')
ireland_df.to_csv(ireland_csv_path, index=False)
print(f"Ireland data saved to: {ireland_csv_path}")

# If you prefer to use actual Ireland datasets, you can try these packages:
print("\nOther packages with potential Ireland data:")
print("- 'maps' package: contains world maps including Ireland")
print("- 'eurostat' package: contains European statistics including Ireland")
print("- 'IrelandData' package: if available, may contain specific Ireland datasets")