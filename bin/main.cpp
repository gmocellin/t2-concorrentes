#include <iostream>
#include <cstring>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mpi.h>
#include <omp.h>

#define NTHREADS 4

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    int rank = 0, nProcessos = 1, tamanho[4];
    MPI_Status status;
    
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    
    Mat imagem, nova_imagem;
    imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    
    int i,j, linha, coluna;
    
    if (strcmp (argv[1], "thread") == 0) {
        MPI_Init(&argc,&argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nProcessos);
        
        if (rank == 0) {
            int tamanhoColuna = imagem.cols/2;
            int tamanhoLinha = imagem.rows/(nProcessos/2);
            int nLinha = nProcessos/2;
            int nColuna = 2;
            int count = 0;
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
        else
            MPI_Recv(tamanho, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        Range rL = Range(tamanho[0],tamanho[1]+tamanho[0]);
        Range rC = Range(tamanho[2],tamanho[3]+tamanho[2]);
        Mat processing = imagem(rL,rC).clone();
        processing.copyTo(nova_imagem);
        
        //#pragma omp parallel
        for (i = 0; i < processing.rows; i++) {
            for (j = 0; j < processing.cols; j++) {
                float R = 0, G = 0, B = 0;
                //#pragma omp parallel reduction(+:R,G,B)
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
        
        if (rank == 0) {
            Mat aux;
            for (i = 1; i < nProcessos; i++) {
                MPI_Recv(tamanho, 4, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                cout << tamanho[1] << " " << tamanho[3] << endl;
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
            char nome[100] = "new_";
            strcat(nome, argv[2]);
            
            cout << nome << endl;
            imwrite(nome,nova_imagem);
            cout << "fim" << endl;
            //free(nome);
        } 
        else {
            MPI_Send(tamanho, 4, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(nova_imagem.data, tamanho[1]*tamanho[3]*imagem.channels(),MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);   
        }
        
        MPI_Finalize();
    }
    else if (strcmp (argv[1], "normal") == 0) {
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
        char nome[100] = "new_";
        strcat(nome, argv[2]);

        cout << nome;
        imwrite(nome,nova_imagem);
        cout << "fim" << endl;
        //free(nome);
    }
    
    return 0;
}
