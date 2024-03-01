import sys
## Broken for now unsure as to why, encrpytion works as expected decrpytion does not
def EncryptDecrypt(inputFile, outputFile, key, encrypt):
    with open(inputFile, 'r') as fin, open(outputFile, 'w') as fout:
        for c in fin.read():
            if c.isalpha():
                offset = ord('A') if c.isupper() else ord('a')
                temp = (ord(c) - offset + key if encrypt else ord(c) - offset - key) % 26 + offset
                fout.write(chr(temp))
            else:
                fout.write(c)

def main():
    inputFile = ""
    outputFile = ""
    key = 0
    encrypt = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-e":
            encrypt = True
        elif sys.argv[i] == "-d":
            encrypt = False
        elif sys.argv[i] == "-k":
            key = int(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == "-i":
            inputFile = sys.argv[i + 1]
            i += 1
        elif sys.argv[i] == "-o":
            outputFile = sys.argv[i + 1]
            i += 1
        i += 1

    if not inputFile or not outputFile:
        raise ValueError("Input and output file names must be provided.")

    if encrypt:
        EncryptDecrypt(inputFile, outputFile, key, True)
        print(f"Encryption completed using key {key}.")
    else:
        EncryptDecrypt(inputFile, outputFile, key, False)
        print(f"Decryption completed using key {key}.")

if __name__ == "__main__":
    main()
