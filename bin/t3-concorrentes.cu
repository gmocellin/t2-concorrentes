#include <iostream>
#include <cstring>
#include <cstdio>
#include <cuda.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define nThreads 1024

using namespace std;
using namespace cv;

__global__ void smoothCuda(unsigned char *imagemC, unsigned char *nova_imagemC, int *cols, int *rows ){

    int posicaoPixel = blockIdx.x * blockDim.x + threadIdx.x;
    int linhaPixel = (int) posicaoPixel/(*cols);
    int colunaPixel = posicaoPixel % (*cols);

    int linha, coluna;
    if(posicaoPixel == 22){
        printf("ANTES:\n");
        printf("posicaoPixel: %d\n", posicaoPixel);
        printf("linhaPixel: %d   colunaPixel: %d\n", linhaPixel, colunaPixel);
        printf("Pixel: B=%u  G=%u  R=%u\n", imagemC[(linhaPixel * (*cols) + colunaPixel)*3],
                                            imagemC[(linhaPixel * (*cols) + colunaPixel)*3+1],
                                            imagemC[(linhaPixel * (*cols) + colunaPixel)*3+2]);
    }
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
        nova_imagemC[(linhaPixel * (*cols) + colunaPixel)*3] = sum[0]/25;
        nova_imagemC[(linhaPixel * (*cols) + colunaPixel)*3 + 1] = sum[1]/25;
        nova_imagemC[(linhaPixel * (*cols) + colunaPixel)*3 + 2] = sum[2]/25;
    }
    if(posicaoPixel == 22){
        printf("DEPOIS:\n");
        printf("posicaoPixel: %d\n", posicaoPixel);
        printf("linhaPixel: %d   colunaPixel: %d\n", linhaPixel, colunaPixel);
        printf("Pixel: B=%u  G=%u  R=%u\n", imagemC[(linhaPixel * (*cols) + colunaPixel)*3],
                                            imagemC[(linhaPixel * (*cols) + colunaPixel)*3+1],
                                            imagemC[(linhaPixel * (*cols) + colunaPixel)*3+2]);
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
    cout << "START" << endl;
    /* imagem       -   imagem a ser processada
     * nova_imagem  -   imagem final 
     */
    Mat imagem, nova_imagem;
    unsigned char *imagemC, *nova_imagemC;
    int *cols, *rows;
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    nBlocks = (int)(imagem.total()/nThreads) + 1; 

    imagem.copyTo(nova_imagem);

    cudaMalloc(&imagemC, imagem.total());
    cudaMalloc(&nova_imagemC, nova_imagem.total());
    cudaMalloc(&cols, sizeof(int));
    cudaMalloc(&rows, sizeof(int));

    cudaMemcpy(nova_imagemC, nova_imagem.data, nova_imagem.total(), cudaMemcpyHostToDevice);
    cudaMemcpy(imagemC, imagem.data, imagem.total(), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, &imagem.cols, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rows, &imagem.rows, sizeof(int), cudaMemcpyHostToDevice);
    printf("linhas: %d   colunas: %d\n", imagem.rows, imagem.cols); 
    printf("rows*cols = %d\n", imagem.rows*imagem.cols);
    printf("BATATA 2\n");
   
    smoothCuda<<<nBlocks,nThreads>>>(imagemC, nova_imagemC, cols, rows);
    
    //printf("TESTE 2 %u\n", nova_imagemC[10001]);
    cudaMemcpy(nova_imagem.data, nova_imagemC, imagem.total(), cudaMemcpyDeviceToHost);
    
    printf("TESTE 3\n");
    // salva a imagem final com outro nome
    char nome[100] = "new_";
    strcat(nome, argv[2]);
    imwrite(nome,nova_imagem);

    cudaFree(imagemC);
    cudaFree(nova_imagemC);
    cudaFree(cols);
    cudaFree(rows);

    cout << "END" << endl;
    return 0;
}
