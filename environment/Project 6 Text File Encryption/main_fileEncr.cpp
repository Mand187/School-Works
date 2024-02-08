#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

using namespace std;

void EncryptDecrypt(const string& inputFile, const string& outputFile, int key, bool encrypt) {
    fstream fin(inputFile, fstream::in);
    fstream fout(outputFile, fstream::out);
    
    char c;
    while (fin >> noskipws >> c) {
        int temp;
        if (encrypt) {
            temp = (c + key);
        } else {
            temp = (c - key);
        }
        fout << (char)temp;
    }
    
    fin.close();
    fout.close();
}

int main(int argc, char* argv[]) {
    string inputFile;
    string outputFile;
    int key = 0;
    bool encrypt = false;

    // Command-line argument handling
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0) {
            encrypt = true;
        } else if (strcmp(argv[i], "-d") == 0) {
            encrypt = false;
        } else if (strcmp(argv[i], "-k") == 0) {
            if (i + 1 < argc) {
                key = atoi(argv[i + 1]);
                i++; // Skip the next argument, as it has been used as the key value
            } else {
                cout << "Error: Missing key value after -k flag." << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                inputFile = argv[i + 1];
                i++; // Skip the next argument, as it has been used as the input file name
            } else {
                cout << "Error: Missing input file name after -i flag." << endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                outputFile = argv[i + 1];
                i++; // Skip the next argument, as it has been used as the output file name
            } else {
                cout << "Error: Missing output file name after -o flag." << endl;
                return 1;
            }
        }
    }

    if (inputFile.empty() || outputFile.empty()) {
        cout << "Error: Input and output file names must be provided." << endl;
        return 1;
    }

    // Perform encryption or decryption
    if (encrypt) {
        EncryptDecrypt(inputFile, outputFile, key, true);
        cout << "Encryption completed using key " << key << "." << endl;
    } else {
        EncryptDecrypt(inputFile, outputFile, key, false);
        cout << "Decryption completed using key " << key << "." << endl;
    }

    return 0;
}
