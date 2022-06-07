#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>


#define MAX_ITERATIONS 100

#define THRESHOLD 1e-4

float *centroidesGlobal;
float *puntosClusterGlobal;
float delta_global = THRESHOLD + 1;
double current_itr_delta_global;
void leerDatos(const char *nombreArchivo, int *N, float **puntos)
{
	FILE *fptr = fopen(nombreArchivo, "r");

	fscanf(fptr, "%d", N);
	*puntos = (float *)malloc(((*N) * 3) * sizeof(float));

	for (int i = 0; i < (*N) * 3; i++)
	{
		int temp;
		fscanf(fptr, "%d", &temp);
		*(*puntos + i) = temp;
	}

	fclose(fptr);
}

void writeClusters(const char *nombreArchivo, int N, float *puntosCluster)
{
	FILE *fptr = fopen(nombreArchivo, "w");

	for (int i = 0; i < N; i++)
	{
		fprintf(fptr, "%f %f %f %f\n", *(puntosCluster + (i * 4)), *(puntosCluster + (i * 4) + 1), *(puntosCluster + (i * 4) + 2), *(puntosCluster + (i * 4) + 3));
	}

	fclose(fptr);
}

void writeCentroids(const char *nombreArchivo, int K, int iteraciones, float *centroides)
{
	FILE *fptr = fopen(nombreArchivo, "w");

	for (int i = 0; i < iteraciones; i++)
	{
		for (int j = 0; j < K; j++)
		{
			fprintf(fptr, "%f %f %f, ", *(centroides + (i * K) + (j * 3)), *(centroides + (i * K) + (j * 3) + 1), *(centroides + (i * K) + (j * 3) + 2));
		}
		fprintf(fptr, "\n");
	}

	fclose(fptr);
}



double findEuclideanDistance(float *pointA, float *pointB)
{
  // Function to find the Euclidean distance between two points in 3 dimensional space
  return sqrt(pow(((double)(*(pointB + 0)) - (double)(*(pointA + 0))), 2) + pow(((double)(*(pointB + 1)) - (double)(*(pointA + 1))), 2) + pow(((double)(*(pointB + 2)) - (double)(*(pointA + 2))), 2));
}

void threadedClustering(int tID, int N, int K, int nHilos, float *puntos, float **puntosCluster, float **centroides, int *iteraciones)
{
  // tID            - The unique ID of the thread which executes this function (read only)
  // N              - Number of data points in the dataset (read only)
  // K              - Number of clusters to be created (read only)
  // nHilos    - No. of threads using which the algorithm should be executed (read only)
  // puntos    - 3 dimensional data points (read only)
  // puntosCluster - 3 dimensional data points along with the ID of the cluster to which they belong (write)
  // centroides      - 3 dimensional coordinate values of the centroides of the K cluster (write)
  // iteraciones - Number of iterations taken to complete the algorithm (write)



  //printf("Tamanio %d, %d, %d", tamanioPorHilo, inicio, fin);

  double minimaDistancia, distanciaActual;
  int *idClusterActual = (int *)malloc(sizeof(int) * N); // The cluster ID to which the data point belong to after each iteration

  int contIteracion = 0;

  while ((delta_global > THRESHOLD) && (contIteracion < MAX_ITERATIONS))
  {
    float *centroideActual = (float *)calloc(K * 3, sizeof(float)); // Coordinates of the centroides which are calculated at the fin of each iteration
    int *contadorCluster = (int *)calloc(K, sizeof(int));              // No. of data points which belongs to each cluster at the fin of each iteration. Initialised to zero
    for (int i = 0; i < N; i++)
    {
      minimaDistancia = __DBL_MAX__; // minimaDistancia is assigned the largest possible double value

      for (int j = 0; j < K; j++)
      {
        distanciaActual = findEuclideanDistance((puntos + (i * 3)), (centroidesGlobal + (contIteracion * K + j) * 3));
        if (distanciaActual < minimaDistancia)
        {
          minimaDistancia = distanciaActual;
          idClusterActual[i - 0] = j;
        }
      }

      contadorCluster[idClusterActual[i - 0]]++;
      centroideActual[idClusterActual[i - 0] * 3] += puntos[(i * 3)];
      centroideActual[idClusterActual[i - 0] * 3 + 1] += puntos[(i * 3) + 1];
      centroideActual[idClusterActual[i - 0] * 3 + 2] += puntos[(i * 3) + 2];
    }


      for (int i = 0; i < K; i++)
      {
        if (contadorCluster[i] == 0)
        {
          // printf("Cluster %d has no data points in it\n", i);
          continue;
        }

        // Update the centroides
        centroidesGlobal[((contIteracion + 1) * K + i) * 3] = centroideActual[(i * 3)] / (float)contadorCluster[i];
        centroidesGlobal[((contIteracion + 1) * K + i) * 3 + 1] = centroideActual[(i * 3) + 1] / (float)contadorCluster[i];
        centroidesGlobal[((contIteracion + 1) * K + i) * 3 + 2] = centroideActual[(i * 3) + 2] / (float)contadorCluster[i];
      }


    // Find delta value after each iteration in all the threads
    double current_delta = 0.0;
    for (int i = 0; i < K; i++)
    {
      current_delta += findEuclideanDistance((centroidesGlobal + (contIteracion * K + i) * 3), (centroidesGlobal + ((contIteracion - 1) * K + i) * 3));
    }

    // Store the largest delta value among all delta values in all the threads
      if (current_delta > current_itr_delta_global)
        current_itr_delta_global = current_delta;


    contIteracion++;

    // Set the global delta value and increment the number of iterations

    delta_global = current_itr_delta_global;
    current_itr_delta_global = 0.0;
    (*iteraciones)++;

  }

  // Update the puntosCluster
  for (int i = 0; i < N; i++)
  {
    puntosClusterGlobal[i * 4] = puntos[i * 3];
    puntosClusterGlobal[i * 4 + 1] = puntos[i * 3 + 1];
    puntosClusterGlobal[i * 4 + 2] = puntos[i * 3 + 2];
    puntosClusterGlobal[i * 4 + 3] = (float)idClusterActual[i - 0];
  }

  // printf("After updating the cluster points %d\n", tID);
}

void kMeans(int N, int K, int nHilos, float *puntos, float **puntosCluster, float **centroides, int *iteraciones)
{
  // 7 arguments
  // N              - Number of data points in the dataset (read only)
  // K              - Number of clusters to be created (read only)
  // nHilos    - No. of threads using which the algorithm should be executed (read only)
  // puntos    - 3 dimensional data points (read only)
  // puntosCluster - 3 dimensional data points along with the ID of the cluster to which they belong (write)
  // centroides      - 3 dimensional coordinate values of the centroides of the K cluster (write)
  // iteraciones - Number of iterations taken to complete the algorithm (write)

  *puntosCluster = (float *)malloc(sizeof(float) * N * 4);
  puntosClusterGlobal = *puntosCluster;

  // calloc intitalizes the values to zero
  centroidesGlobal = (float *)calloc(MAX_ITERATIONS * K * 3, sizeof(float));

  // Assigning the first K data points to be the centroides of the K clusters
  for (int i = 0; i < K; i++)
  {
    centroidesGlobal[(i * 3)] = puntos[(i * 3)];
    centroidesGlobal[(i * 3) + 1] = puntos[(i * 3) + 1];
    centroidesGlobal[(i * 3) + 2] = puntos[(i * 3) + 2];
  }


    int tID = 1;
    printf("Thread no. %d created\n", tID);
    threadedClustering(tID, N, K, nHilos, puntos, puntosCluster, centroides, iteraciones);

  int centroids_size = (*iteraciones + 1) * K * 3;
  *centroides = (float *)calloc(centroids_size, sizeof(float));
  for (int i = 0; i < centroids_size; i++)
  {
    (*centroides)[i] = centroidesGlobal[i];
  }

  printf("Process Completed\n");
  printf("Number of iterations: %d\n", *iteraciones);
  for (int i = 0; i < K; i++)
  {
    printf("Final centroides:\t(%f, %f, %f)\n", *(*centroides + ((*iteraciones) * K) + (i * 3)), *(*centroides + ((*iteraciones) * K) + (i * 3) + 1), *(*centroides + ((*iteraciones) * K) + (i * 3) + 2));
  }
}

int main(int argc, char const *argv[])
{
	if (argc < 6)
	{
		printf("Less no. of command line arguments\n\n");
		return 0;
	}
	else if (argc > 6)
	{
		printf("Too many command line arguments\n\n");
		return 0;
	}

	// Correct no. of command line arguments

	const char *conjuntoDatos = argv[1];
	const int K = atoi(argv[2]);
	const int nHilos = 1;
	const char *archivoPuntosCluster = argv[4];
	const char *archivoCentroides = argv[5];

	int N;								 // Total no. of puntos in the file
	float *puntos;		 // 2D array of 3 dimensional data points
	float *puntosCluster; // 2D array of data points along with the ID of the cluster they belong to
	float *centroides;			 // 2D array of the coordinates of the centroides of clusters
	int iteraciones = 0;

	leerDatos(conjuntoDatos, &N, &puntos);

	clock_t tiempoInicial = clock();
	kMeans(N, K, nHilos, puntos, &puntosCluster, &centroides, &iteraciones);
	clock_t tiempoFinal = clock();

	printf("Tiempo total: %lfs\n", (double) (tiempoFinal - tiempoInicial) / CLOCKS_PER_SEC);

	writeClusters(archivoPuntosCluster, N, puntosCluster);
	writeCentroids(archivoCentroides, K, iteraciones, centroides);

	return 0;
}