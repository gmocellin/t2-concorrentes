#include <iostream>
#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define nThreads 1024

using namespace std;
using namespace cv;

__global__ void smoothCuda(unsigned char *imagemC, unsigned char *nova_imagemC, long unsigned int *cols, long unsigned int *rows ){

    unsigned int posicaoPixel = blockIdx.x * blockDim.x + threadIdx.x;
    int linhaPixel = (int) posicaoPixel/(*cols);
    int colunaPixel = posicaoPixel % (*cols);

    int linha, coluna;
    if(posicaoPixel < (*rows)*(*cols)) {
        int sum[3] = {0, 0, 0};
        for(linha = (linhaPixel - 2); linha <= (linhaPixel + 2); linha++){
            for(coluna = (colunaPixel - 2); coluna <= (colunaPixel + 2); coluna++) {
                if ((linha >= 0 && linha < (*rows)) && (coluna >= 0 && coluna < (*cols))) {
                    sum[0] += imagemC[(linha * (*cols) + coluna)*3];
                    sum[1] += imagemC[(linha * (*cols) + coluna)*3 + 1];
                    sum[2] += imagemC[(linha * (*cols) + coluna)*3 + 2];
                }
            }
        }
        nova_imagemC[posicaoPixel*3] = sum[0]/25;
        nova_imagemC[posicaoPixel*3 + 1] = sum[1]/25;
        nova_imagemC[posicaoPixel*3 + 2] = sum[2]/25;
    }
}

int main(int argc, char *argv[]) {
    /* rank         -   rank das threads em openMPI
     * nProcessos   -   numero de processos disponiveis para executarem openMPI
     * tamanho      -   vetor com os tamanhos de um corte da imagem
     * status       -   status da thread
     */
    int nBlocks;


    // eh necessario receber por parametro o tipo de execucao (thread ou normal) e o arquivo a ser processado, nessa ordem
    if (argc != 2) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    /* imagem       -   imagem a ser processada
     * nova_imagem  -   imagem final 
     */
    Mat imagem, nova_imagem;
    unsigned char *imagemC, *nova_imagemC;
    long unsigned int *cols, *rows;
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    nBlocks = ceil(imagem.total()/nThreads); 

    imagem.copyTo(nova_imagem);
    //printf("imagem.total = %ld   %ld\n", imagem.total(), imagem.total()*3);
    cudaMalloc(&imagemC, 3 * imagem.total() * sizeof(unsigned char));
    cudaMalloc(&nova_imagemC, 3 * nova_imagem.total() * sizeof(unsigned char));
    cudaMalloc(&cols, sizeof(long unsigned int));
    cudaMalloc(&rows, sizeof(long unsigned int));

    cudaMemcpy(nova_imagemC, nova_imagem.data, 3 * nova_imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(imagemC, imagem.data, 3 * imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, &imagem.cols, sizeof(long unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(rows, &imagem.rows, sizeof(long unsigned int), cudaMemcpyHostToDevice);
  
    smoothCuda<<<nBlocks,nThreads>>>(imagemC, nova_imagemC, cols, rows);
    
    cout << cudaGetErrorName(cudaGetLastError()) << endl;    

    cudaMemcpy(nova_imagem.data, nova_imagemC, 3 *  nova_imagem.total() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // salva a imagem final com outro nome
    char nome[100] = "new_";
    strcat(nome, argv[1]);
    imwrite(nome,nova_imagem);

    cudaFree(imagemC);
    cudaFree(nova_imagemC);
    cudaFree(cols);
    cudaFree(rows);

    return 0;
}
