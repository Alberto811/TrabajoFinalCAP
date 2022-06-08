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


double calcularDistanciaEuclidiana(float *puntoA, float *puntoB) {
  // calcular la distancia euclidiana
  return sqrt(pow(((double)(*(puntoB + 0)) - (double)(*(puntoA + 0))), 2) + pow(((double)(*(puntoB + 1)) - (double)(*(puntoA + 1))), 2) + pow(((double)(*(puntoB + 2)) - (double)(*(puntoA + 2))), 2));
}

void ejecucionAlgoritmo(int N, int K, float *puntos, float **puntosCluster, float **centroides, int *iteraciones) {

  double minimaDistancia, distanciaActual;
  //se usa para guardar el id del cluster al que pertenece cada punto en cada momento
  int *idClusterActual = (int *)malloc(sizeof(int) * N);

  int contIteracion = 0;

  while ((deltaGlobal > ERROR) && (contIteracion < ITERACIONES_MAXIMAS)) {
    //aquí se guardan las coordenadas del centroide con el que estemos trabajando en cada iteración
    float *centroideActual = (float *)calloc(K * 3, sizeof(float));
    int *contadorCluster = (int *)calloc(K, sizeof(int));

    for (int i = 0; i < N; i++) {
      //se inicializa la distancia mínima a un valor muy alto
      minimaDistancia = __DBL_MAX__; 

      for (int j = 0; j < K; j++) {
        distanciaActual = calcularDistanciaEuclidiana((puntos + (i * 3)), (centroidesGlobal + (contIteracion * K + j) * 3));

        if (distanciaActual < minimaDistancia) {
          minimaDistancia = distanciaActual;
          idClusterActual[i - 0] = j;
        }
      }

      contadorCluster[idClusterActual[i - 0]]++;
      centroideActual[idClusterActual[i - 0] * 3] += puntos[(i * 3)];
      centroideActual[idClusterActual[i - 0] * 3 + 1] += puntos[(i * 3) + 1];
      centroideActual[idClusterActual[i - 0] * 3 + 2] += puntos[(i * 3) + 2];
    }


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


    // Calculamos el delta que nos dará el grado de convergencia del algoritmo para sabe cuándo parar
    double deltaActualAux = 0.0;
    for (int i = 0; i < K; i++) {
      deltaActualAux += calcularDistanciaEuclidiana((centroidesGlobal + (contIteracion * K + i) * 3), (centroidesGlobal + ((contIteracion - 1) * K + i) * 3));
    }

    // Se guarda el valor de delta mayor
      if (deltaActualAux > deltaActual) {
        deltaActual = deltaActualAux;
      }

    contIteracion++;

    // Actualizamos el valor de delta global
    deltaGlobal = deltaActual;
    deltaActual = 0.0;
    (*iteraciones)++;

  }

  //Actualizamos la asignación de puntos a sus respectivos clusters
  for (int i = 0; i < N; i++) {
    puntosClusterGlobal[i * 4] = puntos[i * 3];
    puntosClusterGlobal[i * 4 + 1] = puntos[i * 3 + 1];
    puntosClusterGlobal[i * 4 + 2] = puntos[i * 3 + 2];
    puntosClusterGlobal[i * 4 + 3] = (float)idClusterActual[i - 0];
  }
}

void kMeans(int N, int K, float *puntos, float **puntosCluster, float **centroides, int *iteraciones) {
  *puntosCluster = (float *)malloc(sizeof(float) * N * 4);
  puntosClusterGlobal = *puntosCluster;

  centroidesGlobal = (float *)calloc(ITERACIONES_MAXIMAS * K * 3, sizeof(float));

  for (int i = 0; i < K; i++) {
    centroidesGlobal[(i * 3)] = puntos[(i * 3)];
    centroidesGlobal[(i * 3) + 1] = puntos[(i * 3) + 1];
    centroidesGlobal[(i * 3) + 2] = puntos[(i * 3) + 2];
  }
  //Ejecutamos todala la lógica del algoritmo
  ejecucionAlgoritmo(N, K, puntos, puntosCluster, centroides, iteraciones);

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
	kMeans(N, K, puntos, &puntosCluster, &centroides, &iteraciones);
	clock_t tiempoFinal = clock();

	printf("Tiempo total: %lfs\n", (double) (tiempoFinal - tiempoInicial) / CLOCKS_PER_SEC);

	return 0;
}