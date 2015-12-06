/*
    Trabalho 3 - CUDA
    Giovane Cunha Mocellin              - 8778382
    Antônio Pedro Lavezzo Mazzarolo     - 8626232

    Implementação utilizando CUDA para realizar smooth em uma imagem.
*/

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//Define de quantas threads serão utilizadas por bloco da GPU
#define nThreads 16

using namespace std;
using namespace cv;

//Funcao que ira realizar o processamento do smooth em paralelo com CUDA
//Eh importante ressaltar que cada thread executara uma linha so da imagem
__global__ void smoothCuda(unsigned char *imagemC, unsigned char *nova_imagemC, int cols, int rows, int canais ){
    
    //Essa linha é muito impoortante, pois eh nela que eh definido qual linha sera processada
    int numLinha = blockIdx.x * blockDim.x + threadIdx.x;
   
    int posicaoPixel, linhaPixel, colunaPixel, linha, coluna;
    
    //Algoritmo para realizacao do smooth(Cada thread executa uma linha)
    //Esse if eh necessario pois pode ser que uma thread ultrapasse o numero de linhas da imagem
    if(numLinha < rows) {
       linhaPixel = numLinha;
        for(posicaoPixel = (numLinha*cols) ; posicaoPixel < ((numLinha*cols)+cols) ; posicaoPixel++){ 
            int sum[3] = {0, 0, 0};
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

    //Leitura da imagem utilizando openCV
    imagem = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    //Verifica se a leitura foi feita com êxito
    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }

    cols = imagem.cols;
    rows = imagem.rows;
    
    //Cáculo do número de blocos a ser alocado na GPU com base na quantidade de linhas da imagem e no número de threads definido
    nBlocks = ceil(imagem.rows/nThreads); 
 
    //Cópida da imagem original para a nova imagem
    imagem.copyTo(nova_imagem);
    
    //Alocacao de memoria da GPU com base no tamanho das imagens
    cudaMalloc(&imagemC, imagem.channels() * imagem.total() * sizeof(unsigned char));
    cudaMalloc(&nova_imagemC, imagem.channels() * nova_imagem.total() * sizeof(unsigned char));

    //Cópia das imagens para a GPU
    cudaMemcpy(nova_imagemC, nova_imagem.data, imagem.channels() * nova_imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(imagemC, imagem.data, imagem.channels() * imagem.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //Chamada do kernel para execucao do smooth em paralelo usando CUDA
    //nBlocks define o numero de blocos a ser usado na GPU e nthreads define o numero de threads
    //eh passado como parametro a imagem origina e a nova, o numero de colunas, de linhas e a quantidade de canais da imagem
    //esses dados sao necessarios para fazer o smooth
    smoothCuda<<<nBlocks,nThreads>>>(imagemC, nova_imagemC, cols, rows, imagem.channels());

    //Copia a nova imagem de volta para a main. Em nova imagem esta amazenado o resultado do smooth
    cudaMemcpy(nova_imagem.data, nova_imagemC, imagem.channels() * nova_imagem.total() * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // salva a imagem final com outro nome
    char nome[100] = "new_";
    strcat(nome, argv[1]);
    imwrite(nome,nova_imagem);

    //Libera memoria da GPU
    cudaFree(imagemC);
    cudaFree(nova_imagemC);

    return 0;
}
