'''
SMILES (Simplified Molecular Input Line Entry System) is a compact text notation for representing chemical structures. 

In RDKit, a popular cheminformatics library, 
SMILES strings are one of the primary ways to input and manipulate molecular data.

Basic SMILES Syntax: SMILES uses simple rules to encode molecular structures:
# Atoms: Represented by their element symbols (C for carbon, O for oxygen, N for nitrogen, etc.)
# Bonds: Single bonds are implicit, double bonds use =, triple bonds use #, aromatic bonds use lowercase letters
# Branches: Indicated with parentheses ()
# Rings: Numbered to show where rings close (digits like 1, 2, 3)
# Charges: + or - symbols after atoms

Examples
# CCO - ethanol (two carbons connected to an oxygen)
# c1ccccc1 - benzene (aromatic six-membered ring)
# CC(=O)O - acetic acid (has a branching carboxylic acid group)
# CC(C)C - isobutane (branched alkane)
'''

###################
## Install rdkit ##
###################

# Run in terminal: pip3 install rdkit
# Or             : conda install -c conda-forge rdkit

# Verification:
import rdkit

methane = rdkit.Chem.MolFromSmiles("C")
methane

#---------
## Show image in pop-up window
#---------

img = rdkit.Chem.Draw.MolToImage(methane) # Create an image from a molecule object
img.show() # Show the image in a pop-up window (for vscode script)

######################
## Drawing molecule ##
######################

import rdkit
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs

#------
## without atom index
#------

# A kinase inhibitor
mol = rdkit.Chem.MolFromSmiles("C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65")
mol

#------
## with atom index
#------

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

mol_with_atom_index(mol)

#------
## with atom index (Simpler way)
#------
'''produces a similar image to the example above, the difference being that the atom indices are now near the atom, rather than at the atom position.'''

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True

mol = rdkit.Chem.MolFromSmiles("C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65")
mol

##########################
## Include a Bond Index ##
##########################

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addBondIndices = True
IPythonConsole.drawOptions.addAtomIndices = False # Turn off AtomIndices for better visiblity

mol = rdkit.Chem.MolFromSmiles("C1CC2=C3C(=CC=C2)C(=CN3C1)[C@H]4[C@@H](C(=O)NC4=O)C5=CNC6=CC=CC=C65")
mol