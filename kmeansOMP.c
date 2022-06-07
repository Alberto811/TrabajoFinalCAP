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
void leerDatos(const char *nombreArchivo, int *N, float **puntos) {
	FILE *fptr = fopen(nombreArchivo, "r");

	fscanf(fptr, "%d", N);
	*puntos = (float *)malloc(((*N) * 3) * sizeof(float));
	for (int i = 0; i < (*N) * 3; i++) {
		int temp;
		fscanf(fptr, "%d", &temp);
		*(*puntos + i) = temp;
	}

	fclose(fptr);
}

void escribirArchivoClusters(const char *nombreArchivo, int N, float *puntosCluster) {
	FILE *fptr = fopen(nombreArchivo, "w");

	for (int i = 0; i < N; i++) {
		fprintf(fptr, "%f %f %f %f\n", *(puntosCluster + (i * 4)), *(puntosCluster + (i * 4) + 1), *(puntosCluster + (i * 4) + 2), *(puntosCluster + (i * 4) + 3));
	}

	fclose(fptr);
}

void escribirArchivoCentroides(const char *nombreArchivo, int K, int iteraciones, float *centroides) {
	FILE *fptr = fopen(nombreArchivo, "w");

	for (int i = 0; i < iteraciones; i++) {
		for (int j = 0; j < K; j++) {
			fprintf(fptr, "%f %f %f, ", *(centroides + (i * K) + (j * 3)), *(centroides + (i * K) + (j * 3) + 1), *(centroides + (i * K) + (j * 3) + 2));
		}
		fprintf(fptr, "\n");
	}

	fclose(fptr);
}



double calcularDistanciaEuclidiana(float *puntoA, float *puntoB) {
  // calcular la distancia euclidiana
  return sqrt(pow(((double)(*(puntoB + 0)) - (double)(*(puntoA + 0))), 2) + pow(((double)(*(puntoB + 1)) - (double)(*(puntoA + 1))), 2) + pow(((double)(*(puntoB + 2)) - (double)(*(puntoA + 2))), 2));
}

void ejecucionAlgoritmo(int hiloID, int N, int K, int nHilos, float *puntos, float **puntosCluster, float **centroides, int *iteraciones) {

  printf("Hilo:  %d\n", hiloID);
  int longitudHilo = N / nHilos;
  int inicio = hiloID * longitudHilo;
  int fin = inicio + longitudHilo;

  if (fin > N)
  {
    fin = N;
    longitudHilo = inicio - fin;
  }


  double minimaDistancia, distanciaActual;
  //se usa para guardar el id del cluster al que pertenece cada punto en cada momento
  int *idClusterActual = (int *)malloc(sizeof(int) * longitudHilo);

  int contIteracion = 0;

  while ((deltaGlobal > ERROR) && (contIteracion < ITERACIONES_MAXIMAS)) {
    //aquí se guardan las coordenadas del centroide con el que estemos trabajando en cada iteración
    float *centroideActual = (float *)calloc(K * 3, sizeof(float));
    int *contadorCluster = (int *)calloc(K, sizeof(int));

    for (int i = inicio; i < fin; i++) {
      //se inicializa la distancia mínima a un valor muy alto
      minimaDistancia = __DBL_MAX__; 

      for (int j = 0; j < K; j++) {
        distanciaActual = calcularDistanciaEuclidiana((puntos + (i * 3)), (centroidesGlobal + (contIteracion * K + j) * 3));

        if (distanciaActual < minimaDistancia) {
          minimaDistancia = distanciaActual;
          idClusterActual[i - inicio] = j;
        }
      }

      contadorCluster[idClusterActual[i - inicio]]++;
      centroideActual[idClusterActual[i - inicio] * 3] += puntos[(i * 3)];
      centroideActual[idClusterActual[i - inicio] * 3 + 1] += puntos[(i * 3) + 1];
      centroideActual[idClusterActual[i - inicio] * 3 + 2] += puntos[(i * 3) + 2];
    }

    #pragma omp critical
      {
        for (int i = 0; i < K; i++) {
          if (contadorCluster[i] == 0) {
            // aquí sólo entra si el cluster no tiene puntos
            continue;
          }

          //Actualizamos los centroides
          centroidesGlobal[((contIteracion + 1) * K + i) * 3] = centroideActual[(i * 3)] / (float)contadorCluster[i];
          centroidesGlobal[((contIteracion + 1) * K + i) * 3 + 1] = centroideActual[(i * 3) + 1] / (float)contadorCluster[i];
          centroidesGlobal[((contIteracion + 1) * K + i) * 3 + 2] = centroideActual[(i * 3) + 2] / (float)contadorCluster[i];
        }
      }


    // Calculamos el delta que nos dará el grado de convergencia del algoritmo para sabe cuándo parar
    double deltaActualAux = 0.0;
    for (int i = 0; i < K; i++) {
      deltaActualAux += calcularDistanciaEuclidiana((centroidesGlobal + (contIteracion * K + i) * 3), (centroidesGlobal + ((contIteracion - 1) * K + i) * 3));
    }

    #pragma omp barrier
      {
        // Se guarda el valor de delta mayor
        if (deltaActualAux > deltaActual) {
          deltaActual = deltaActualAux;
        }
      }
    #pragma omp critical 
    {
    contIteracion++;
    }

    #pragma omp master
    {
      // Actualizamos el valor de delta global
      deltaGlobal = deltaActual;
      deltaActual = 0.0;
      (*iteraciones)++;
    }
  }

  //Actualizamos la asignación de puntos a sus respectivos clusters
  for (int i = inicio; i < fin; i++) {
    puntosClusterGlobal[i * 4] = puntos[i * 3];
    puntosClusterGlobal[i * 4 + 1] = puntos[i * 3 + 1];
    puntosClusterGlobal[i * 4 + 2] = puntos[i * 3 + 2];
    puntosClusterGlobal[i * 4 + 3] = (float)idClusterActual[i - inicio];
  }
}

void kMeans(int N, int K, int nHilos, float *puntos, float **puntosCluster, float **centroides, int *iteraciones) {
  *puntosCluster = (float *)malloc(sizeof(float) * N * 4);
  puntosClusterGlobal = *puntosCluster;

  centroidesGlobal = (float *)calloc(ITERACIONES_MAXIMAS * K * 3, sizeof(float));

  for (int i = 0; i < K; i++) {
    centroidesGlobal[(i * 3)] = puntos[(i * 3)];
    centroidesGlobal[(i * 3) + 1] = puntos[(i * 3) + 1];
    centroidesGlobal[(i * 3) + 2] = puntos[(i * 3) + 2];
  }
  omp_set_num_threads(nHilos);

  #pragma omp parallel 
  {
  int hiloID = omp_get_thread_num();
  //Ejecutamos todala la lógica del algoritmo de forma paralela
  ejecucionAlgoritmo(hiloID, N, K, nHilos, puntos, puntosCluster, centroides, iteraciones);
  }

  int tamanioCentroides = (*iteraciones + 1) * K * 3;
  *centroides = (float *)calloc(tamanioCentroides, sizeof(float));

  for (int i = 0; i < tamanioCentroides; i++) {
    (*centroides)[i] = centroidesGlobal[i];
  }

  printf("Ejecución terminada\n");
  printf("Número de iteraciones: %d\n", *iteraciones);
  for (int i = 0; i < K; i++) {
    printf("Centroides 3 dimensiones:\t(%f, %f, %f)\n", *(*centroides + ((*iteraciones) * K) + (i * 3)), *(*centroides + ((*iteraciones) * K) + (i * 3) + 1), *(*centroides + ((*iteraciones) * K) + (i * 3) + 2));
  }
}

int main(int argc, char const *argv[]) {

	const char *conjuntoDatos = argv[1];
	const int K = atoi(argv[2]);
	const int nHilos = atoi(argv[3]);
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

	double tiempoInicial = omp_get_wtime();
	kMeans(N, K, nHilos, puntos, &puntosCluster, &centroides, &iteraciones);
	double tiempoFinal = omp_get_wtime();

	printf("Tiempo total: %lfs\n", tiempoFinal - tiempoInicial);

	escribirArchivoClusters(archivoPuntosCluster, N, puntosCluster);
	escribirArchivoCentroides(archivoCentroides, K, iteraciones, centroides);

	return 0;
}