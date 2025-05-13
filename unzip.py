from zipfile import ZipFile
with ZipFile("archive.zip", 'r') as zObject: 
	zObject.extractall(path="./datasets/A")