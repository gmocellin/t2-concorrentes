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

__global__ void smoothCuda(unsigned char *imagemC, unsigned char *nova_imagemC, int cols, int rows, int canais ){

    int numLinha = blockIdx.x * blockDim.x + threadIdx.x;
   
    int posicaoPixel, linhaPixel, colunaPixel, linha, coluna;
    //printf("posPixel = %d\n", posicaoPixel);
    if(numLinha < rows*cols) {
        int sum[3] = {0, 0, 0};
        for(posicaoPixel = numLinha ; posicaoPixel < (numLinha+cols) ; posicaoPixel++){ 
            linhaPixel = (int)(posicaoPixel/cols);
            colunaPixel = posicaoPixel % cols;
            for(linha = (linhaPixel - 2); linha <= (linhaPixel + 2); linha++){
                for(coluna = (colunaPixel - 2); coluna <= (colunaPixel + 2); coluna++) {
                    if ((linha >= 0 && linha < rows) && (coluna >= 0 && coluna < cols)) {
                        if( canais == 1){
                            sum[0] += imagemC[(linha * cols + coluna)];
                        } else {    
                            sum[0] += imagemC[(linha * cols + coluna)*3];
                            sum[1] += imagemC[(linha * cols + coluna)*3 + 1];
                            sum[2] += imagemC[(linha * cols + coluna)*3 + 2];
                        }
                    }
                }
            }
       
            if( canais == 1){
                nova_imagemC[posicaoPixel] = sum[0]/25;
            } else {
                nova_imagemC[posicaoPixel*3] = sum[0]/25;
                nova_imagemC[posicaoPixel*3 + 1] = sum[1]/25;
                nova_imagemC[posicaoPixel*3 + 2] = sum[2]/25;
            }
        }
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
    int cols, rows;

    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }

    cols = imagem.cols;
    rows = imagem.rows;
    nBlocks = ceil(imagem.rows/nThreads); 
 
    cout << imagem.channels() << endl;
    cout << nBlocks << endl;
    printf("imagem.total = %ld  cols = %d  rows =%d\n", imagem.total(), cols, rows);
    
    imagem.copyTo(nova_imagem);
    cudaMalloc(&imagemC, imagem.channels() * imagem.total() * sizeof(unsigned char));
    cudaMalloc(&nova_imagemC, imagem.channels() * nova_imagem.total() * sizeof(unsigned char));

    cudaMemcpy(nova_imagemC, nova_imagem.data, imagem.channels() * nova_imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(imagemC, imagem.data, imagem.channels() * imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    smoothCuda<<<nBlocks,nThreads>>>(imagemC, nova_imagemC, cols, rows, imagem.channels());

    cout << cudaGetErrorName(cudaGetLastError()) << endl;    

    cudaMemcpy(nova_imagem.data, nova_imagemC, imagem.channels() * nova_imagem.total() * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // salva a imagem final com outro nome
    char nome[100] = "new_";
    strcat(nome, argv[1]);
    imwrite(nome,nova_imagem);

    cudaFree(imagemC);
    cudaFree(nova_imagemC);

    return 0;
}
