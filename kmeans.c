#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>


#define ITERACIONES_MAXIMAS 100

#define ERROR 1e-4

float *centroidesGlobal;
float *puntosClusterGlobal;
float deltaGlobal = ERROR + 1;
double deltaActual;
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

void escribirArchivoClusters(const char *nombreArchivo, int N, float *puntosCluster)
{
	FILE *fptr = fopen(nombreArchivo, "w");

	for (int i = 0; i < N; i++)
	{
		fprintf(fptr, "%f %f %f %f\n", *(puntosCluster + (i * 4)), *(puntosCluster + (i * 4) + 1), *(puntosCluster + (i * 4) + 2), *(puntosCluster + (i * 4) + 3));
	}

	fclose(fptr);
}

void escribirArchivoCentroides(const char *nombreArchivo, int K, int iteraciones, float *centroides)
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



double calcularDistanciaEuclidiana(float *puntoA, float *puntoB)
{
  // Function to find the Euclidean distance between two points in 3 dimensional space
  return sqrt(pow(((double)(*(puntoB + 0)) - (double)(*(puntoA + 0))), 2) + pow(((double)(*(puntoB + 1)) - (double)(*(puntoA + 1))), 2) + pow(((double)(*(puntoB + 2)) - (double)(*(puntoA + 2))), 2));
}

void ejecucionAlgoritmo(int N, int K, int nHilos, float *puntos, float **puntosCluster, float **centroides, int *iteraciones)
{

  double minimaDistancia, distanciaActual;
  int *idClusterActual = (int *)malloc(sizeof(int) * N); // The cluster ID to which the data point belong to after each iteration

  int contIteracion = 0;

  while ((deltaGlobal > ERROR) && (contIteracion < ITERACIONES_MAXIMAS))
  {
    float *centroideActual = (float *)calloc(K * 3, sizeof(float)); // Coordinates of the centroides which are calculated at the fin of each iteration
    int *contadorCluster = (int *)calloc(K, sizeof(int));              // No. of data points which belongs to each cluster at the fin of each iteration. Initialised to zero
    for (int i = 0; i < N; i++)
    {
      minimaDistancia = __DBL_MAX__; // minimaDistancia is assigned the largest possible double value

      for (int j = 0; j < K; j++)
      {
        distanciaActual = calcularDistanciaEuclidiana((puntos + (i * 3)), (centroidesGlobal + (contIteracion * K + j) * 3));
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
      current_delta += calcularDistanciaEuclidiana((centroidesGlobal + (contIteracion * K + i) * 3), (centroidesGlobal + ((contIteracion - 1) * K + i) * 3));
    }

    // Store the largest delta value among all delta values in all the threads
      if (current_delta > deltaActual)
        deltaActual = current_delta;


    contIteracion++;

    // Set the global delta value and increment the number of iterations

    deltaGlobal = deltaActual;
    deltaActual = 0.0;
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
  centroidesGlobal = (float *)calloc(ITERACIONES_MAXIMAS * K * 3, sizeof(float));

  // Assigning the first K data points to be the centroides of the K clusters
  for (int i = 0; i < K; i++)
  {
    centroidesGlobal[(i * 3)] = puntos[(i * 3)];
    centroidesGlobal[(i * 3) + 1] = puntos[(i * 3) + 1];
    centroidesGlobal[(i * 3) + 2] = puntos[(i * 3) + 2];
  }


  ejecucionAlgoritmo(N, K, nHilos, puntos, puntosCluster, centroides, iteraciones);

  int tamanioCentroides = (*iteraciones + 1) * K * 3;
  *centroides = (float *)calloc(tamanioCentroides, sizeof(float));
  for (int i = 0; i < tamanioCentroides; i++)
  {
    (*centroides)[i] = centroidesGlobal[i];
  }

  printf("Ejecución terminada\n");
  printf("Número de iteraciones: %d\n", *iteraciones);
  for (int i = 0; i < K; i++)
  {
    printf("Centroides 3 dimensiones:\t(%f, %f, %f)\n", *(*centroides + ((*iteraciones) * K) + (i * 3)), *(*centroides + ((*iteraciones) * K) + (i * 3) + 1), *(*centroides + ((*iteraciones) * K) + (i * 3) + 2));
  }
}

int main(int argc, char const *argv[])
{

	const char *conjuntoDatos = argv[1];
	const int K = atoi(argv[2]);
	const int nHilos = 1;
	const char *archivoPuntosCluster = argv[4];
	const char *archivoCentroides = argv[5];

	//total de puntos de datos
  int N;
  //puntos sacados de los datos de entrada
	float *puntos;
  //puntos con el ID del cluster al que pertenece
	float *puntosCluster; // 2D array of data points along with the ID of the cluster they belong to
  //coordenadas de los centroides
	float *centroides;
  //iteraciones
	int iteraciones = 0;

	leerDatos(conjuntoDatos, &N, &puntos);

	clock_t tiempoInicial = clock();
	kMeans(N, K, nHilos, puntos, &puntosCluster, &centroides, &iteraciones);
	clock_t tiempoFinal = clock();

	printf("Tiempo total: %lfs\n", (double) (tiempoFinal - tiempoInicial) / CLOCKS_PER_SEC);

	escribirArchivoClusters(archivoPuntosCluster, N, puntosCluster);
	escribirArchivoCentroides(archivoCentroides, K, iteraciones, centroides);

	return 0;
}