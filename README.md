# TrabajoFinalCAP

kmeans.c (SECUENCIAL)

gcc kmeans.c -o kmeans -lm

./kmeans datos/longitud10000.txt 25

kmeans.c (PARALELIZADO)

gcc kmeansOMP.c -fopenmp -o kmeansOMP -lm

./kmeansOMP datos/longitud10000.txt 25 8
