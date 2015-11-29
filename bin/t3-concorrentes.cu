#include <iostream>
#include <cstring>
#include <cstdio>
#include <cuda.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

__global__ void smoothCuda(){

    for (i = 0; i < imagem.rows; i++) {
        for (j = 0; j < imagem.cols; j++) {
            float sum[3] = {0, 0, 0};
            for(linha = (i - 2); linha <= (i + 2); linha++){
                for(coluna = (j - 2); coluna <= (j + 2); coluna++) {
                    if ((linha >= 0 && linha < imagem.rows) && (coluna >= 0 && coluna < imagem.cols)) {
                        sum[0] += imagem.at<Vec3b>(linha,coluna)[0];
                        sum[1] += imagem.at<Vec3b>(linha,coluna)[1];
                        sum[2] += imagem.at<Vec3b>(linha,coluna)[2];
                    }
                }
            }
            nova_imagem.at<Vec3b>(i,j)[0] = sum[0]/25;
            nova_imagem.at<Vec3b>(i,j)[1] = sum[1]/25;
            nova_imagem.at<Vec3b>(i,j)[2] = sum[2]/25;
        }
    }


}

__global__ void smoothCuda( Mat *imagemC, Mat *nova_imagemC ){

    int posicaoPixel = blockIdx.x * blockDim.x + threadIdx.x;
    int linhaPixel = (int) posicaoPixel;
    int colunaPixel = posicaoPixel % imagemC.cols;

    int linha, coluna;
    if(posicaoPixel < (imagem.rows)*(imagem.cols)) {
        for(linha = (linhaPixel - 2); linha <= (linhaPixel + 2); linha++){
            for(coluna = (colunaPixel - 2); coluna <= (colunaPixel + 2); coluna++) {
                if ((linha >= 0 && linha < imagemC.rows) && (coluna >= 0 && coluna < imagemC.cols)) {
                    sum[0] += imagemC.at<Vec3b>(linha,coluna)[0];
                    sum[1] += imagemC.at<Vec3b>(linha,coluna)[1];
                    sum[2] += imagemC.at<Vec3b>(linha,coluna)[2];
                }
            }
        }
        nova_imagemC.at<Vec3b>(i,j)[0] = sum[0]/25;
        nova_imagemC.at<Vec3b>(i,j)[1] = sum[1]/25;
        nova_imagemC.at<Vec3b>(i,j)[2] = sum[2]/25;
    }
}

int main(int argc, char *argv[]) {
    /* rank         -   rank das threads em openMPI
     * nProcessos   -   numero de processos disponiveis para executarem openMPI
     * tamanho      -   vetor com os tamanhos de um corte da imagem
     * status       -   status da thread
     */
    int rank = 0, nThreads = 1024, nBlocks, tamanho[4];


    // eh necessario receber por parametro o tipo de execucao (thread ou normal) e o arquivo a ser processado, nessa ordem
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }

    /* imagem       -   imagem a ser processada
     * nova_imagem  -   imagem final 
     */
    Mat imagem, nova_imagem, *imagemC, *nova_imagemC;
    imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    nBlocks = (int)((imagem.rows*imagem.cols)/nThreads) + 1; 

    imagem.copyTo(nova_imagem);

    cudaMalloc((void**)&imagemC, sizeof(imagem));
    cudaMalloc((void**)&nova_imagemC, sizeof(nova_imagem));

    cudaMemcpy(nova_imagemC, nova_imagem, sizeof(nova_imagem), cudaMemcpyHostToDevice);
    cudaMemcpy(imagemC, imagem, sizeof(imagem), cudaMemcpyHostToDevice);

    smoothCuda<<<nBlocks,nThreads>>>(imagemC, nova_imagemC);

    cudaMemcpy(nova_imagem, nova_imagemC, sizeof(nova_imageC), cudaMemcpyDeviceToHost);

    // salva a imagem final com outro nome
    char nome[100] = "new_";
    strcat(nome, argv[2]);
    imwrite(nome,nova_imagem);

    cudaFree(imagemC);
    cudaFree(nova_imagemC);

    return 0;
}
