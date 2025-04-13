# SpeedRsa
Иногда скорость генерации выше чем у openssl
super fast analog openssl rsa-2048
gcc -O3 -march=native -mtune=native -pthread rsa_keygen.c -lgmp -lssl -lcrypto -o rsa_keygen -Wno-deprecated-declarations
