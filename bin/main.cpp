#include <iostream>
#include <cstring>
#include <cstdio>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <mpi.h>
#include <omp.h>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    /* rank         -   rank das threads em openMPI
     * nProcessos   -   numero de processos disponiveis para executarem openMPI
     * tamanho      -   vetor com os tamanhos de um corte da imagem
     * status       -   status da thread
     */
    int rank = 0, nProcessos = 1, tamanho[4];
    MPI_Status status;
    
    // eh necessario receber por parametro o tipo de execucao (thread ou normal) e o arquivo a ser processado, nessa ordem
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    
    /* imagem       -   imagem a ser processada
     * nova_imagem  -   imagem final 
     */
    Mat imagem, nova_imagem;
    imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    
    // auxiliares de for
    int i,j, linha, coluna;
    
    // processamento paralelo
    if (strcmp (argv[1], "thread") == 0) {
        // pega as informacoes do processo do openMPI
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nProcessos);
        
        // processo principal
        if (rank == 0) {
            /* tamanhoColuna    -   largura do bloco cortado
             * tamanhoLinha     -   altura do bloco cortado
             * nLinha           -   numero de blocos por linha
             * nColuna          -   numero de blocos por coluna
             * count            -   contador de processos
             */
            int tamanhoColuna = imagem.cols/2;
            int tamanhoLinha = imagem.rows/(nProcessos/2);
            int nLinha = nProcessos/2;
            int nColuna = 2;
            int count = 0;
            // separa a imagem em blocos e envia a area do bloco para o processo designado
            for (i = 0; i < nLinha; i++) {
                tamanho[0] = tamanhoLinha*i;
                if (tamanhoLinha*(i+1) == ((imagem.rows+1) - (nProcessos/2)))
                    tamanho[1] = ((tamanhoLinha-1)+(nProcessos/2));
                else
                    tamanho[1] = tamanhoLinha;
                for (j = 0; j < nColuna; j++) {
                    if ((i == 0) && (j == 0))
                        continue;
                    count++;
                    tamanho[2] = tamanhoColuna*j;
                    if (tamanhoColuna*(j+1) == (imagem.cols - 1))
                        tamanho[3] = tamanhoColuna+1;
                    else
                        tamanho[3] = tamanhoColuna;

                    MPI_Send(tamanho, 4, MPI_INT, count, 0, MPI_COMM_WORLD);
                }
            }
            tamanho[0] = 0;
            tamanho[1] = tamanhoLinha;
            tamanho[2] = 0;
            tamanho[3] = tamanhoColuna;
        }
        // processos secundarios recebem o tamanho do bloco
        else
            MPI_Recv(tamanho, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        /* rL           -   range do bloco em linha
         * rC           -   range do bloco em coluna
         * processing   -   bloco sendo processado
         */
        Range rL = Range(tamanho[0],tamanho[1]+tamanho[0]);
        Range rC = Range(tamanho[2],tamanho[3]+tamanho[2]);
        nova_imagem = imagem(rL,rC).clone();
        
        // paraleliza o for em openMP
        #pragma omp parallel for
        // faz o smooth no bloco da imagem
        for (i = 0; i < nova_imagem.rows; i++) {
            for (j = 0; j < nova_imagem.cols; j++) {
                /* RGB      -   canais de cores da imagem (Red, Green e Blue, respectivamente) */
                float R = 0, G = 0, B = 0;
                // paraleliza o for em openMP a partir do incremento de R
                #pragma omp parallel for reduction(+:R)
                for(linha = (i - 2); linha <= (i + 2); linha++){
                    for(coluna = (j - 2); coluna <= (j + 2); coluna++) {
                        if ((tamanho[0]+linha >= 0 && tamanho[0]+linha < imagem.rows) 
                                && (tamanho[2]+coluna >= 0 && tamanho[2]+coluna < imagem.cols)) {
                            B += imagem.at<Vec3b>(tamanho[0]+linha,tamanho[2]+coluna)[0];
                            G += imagem.at<Vec3b>(tamanho[0]+linha,tamanho[2]+coluna)[1];
                            R += imagem.at<Vec3b>(tamanho[0]+linha,tamanho[2]+coluna)[2];
                        }
                    }
                }
                nova_imagem.at<Vec3b>(i,j)[0] = B/25;
                nova_imagem.at<Vec3b>(i,j)[1] = G/25;
                nova_imagem.at<Vec3b>(i,j)[2] = R/25;
            }
        }
        
        // processo principal concatena os blocos da imagem recebidos de outros processos
        if (rank == 0) {
            /* aux      -   matriz auxiliar para concatenacao da imagem final */
            Mat aux;
            for (i = 1; i < nProcessos; i++) {
                MPI_Recv(tamanho, 4, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                /* pedaco       -   pedaco da imagem recebido de outro processo */
                Mat pedaco(tamanho[1], tamanho[3], nova_imagem.type());
                MPI_Recv(pedaco.data, tamanho[1]*tamanho[3]*imagem.channels(), MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, &status);
                
                if ((i%2) == 0)
                    vconcat(nova_imagem, pedaco, nova_imagem);
                else
                    if (i == 1)
                        pedaco.copyTo(aux);
                    else
                        vconcat(aux, pedaco, aux);
            }
            
            hconcat(nova_imagem, aux, nova_imagem);
            
            // salva a imagem final com outro nome
            char nome[100] = "new_";
            strcat(nome, argv[2]);
            imwrite(nome,nova_imagem);
        } 
        // processos secundarios enviam o resultado do smooth junto do tamanho do bloco
        else {
            MPI_Send(tamanho, 4, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(nova_imagem.data, tamanho[1]*tamanho[3]*imagem.channels(),MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);   
        }
        
        // finaliza os processos do openMPI
        MPI_Finalize();
    }
    // processamento sequencial
    else if (strcmp (argv[1], "normal") == 0) {
        // funcao smooth
        imagem.copyTo(nova_imagem);
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
        
        // salva a imagem final com outro nome
        char nome[100] = "new_";
        strcat(nome, argv[2]);
        imwrite(nome,nova_imagem);
    }
    
    return 0;
}
