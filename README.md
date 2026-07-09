# Double-sided-Deck-U-rib-SCF-GUI
#This GUI provides users with rapid predictions of the effective notch stress concentration factors (SCF<sub>ENS</sub>) for double-sided deck-to-U-rib welded joints in orthotropic steel decks. Users simply input the required geometric parameters as prompted on the interface to obtain SCF<sub>ENS</sub> at both the internal and external weld toes.

The interface offers two prediction channels:
- For **SCF<sub>ENS-ex</sub>** (external weld toe), the original real dataset-based stacking model integrating CatBoost, GBDT and ANN is adopted;
- For **SCF<sub>ENS-in</sub>** (internal weld toe), the augmented dataset-based simple averaging model integrating XGBoost and GBDT is adopted.
Users can select the desired target according to their needs.

Developed by: College of Civil Engineering, Nanjing Forestry University. 
