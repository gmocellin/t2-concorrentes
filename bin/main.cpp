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
    int rank = 0, numero_de_processos = 1, tamanho[3];
    MPI_Status status;
    
    if (argc != 3) {
        cout << "Quantidade de parametros invalida." << endl;
        return -1;
    }
    
    Mat imagem, nova_imagem;
    if (rank = 0)
        imagem = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    
    if(!imagem.data) {
        cout <<  "Imagem nao encontrada." << endl;
        return -1;
    }
    
    imagem.copyTo(nova_imagem);
    
    int i,j, linha, coluna;
    
    if (strcmp (argv[1], "thread") == 0) {
        if (rank == 0) {
            MPI_Init(&argc,&argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            
            MPI_Comm_size(MPI_COMM_WORLD, &nProcessos);
            
            int tamanhoPedacos = imagem.rows/nProcessos;
            for (i = 1; i < nProcessos; i++) {
                int area = tamanhoPedacos*i;
                
                Rect bloco = Rect(0, area, imagem.cols, tamanhoPedacos);
                Mat pedaco = imagem(bloco);
                
                tamanho[0] = pedaco.size().height;
                tamanho[1] = pedaco.size().width;
                tamanho[2] = pedaco.type();
                
                MPI_Send(tamanho, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
                
                MPI_Send(pedaco.data, tamanho[0]*tamanho[1]*pedaco.channels(), MPI_UNSIGNED_CHAR, i, 1, MPI_COMM_WORLD);
            }
            
            imagem = imagem(Rect(0, 0, imagem.cols, tamanhoPedacos));
        }
        else {
            MPI_Recv(tamanho, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            imagem = Mat(tamanho[0], tamanho[1], tamanho[2]);
            
            MPI_Recv(imagem.data, tamanho[0]*tamanho[1]*imagem.channels(), MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD, &status);
        }
        
        #pragma omp parallel
        for (i = 0; i < imagem.rows; i++) {
            for (j = 0; j < imagem.cols; j++) {
                float sum[3] = {0, 0, 0};
                #pragma omp parallel reduction(+:sum)
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
        
        if (rank == 0) {
            final_image = destiny_image(Rect(3, 3, destiny_image.cols - 1, destiny_image.rows - 3));
            for (i = 1; i < nProcessos; i++) {
                MPI_Recv(tamanho, 3, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
                Mat pedaco(tamanho[0], tamanho[1], tamanho[2]);
                MPI_Recv(pedaco.data, tamanho[0]*tamanho[1]*imagem.channels(), MPI_UNSIGNED_CHAR, i, 3, MPI_COMM_WORLD, &status);
                
                Rect region_of_interest = Rect(3, 3, piece.cols - 1, piece.rows - 3);
                pedaco = pedaco(region_of_interest);
                
                vconcat(final_image, piece, final_image);
            } 
        } 
        else {
            MPI_Send(tamanho, 3, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(destiny_image.data, tamanho[0]*tamanho[1]*destiny_image.channels(), MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD);   
        }
        
        MPI_Finalize();
        char nome[100] = "new_";
        strcat(nome, argv[2]);
        //char *nome = (char *) malloc (100 * sizeof(char)); 
        //strncpy(nome,argv[2],(strlen(argv[2])-4));
        //strcat(nome,"_new.jpg");
        
        cout << nome;
        imwrite(nome,nova_imagem);
        cout << "fim" << endl;
        //free(nome);
    }
    else if (strcmp (argv[1], "normal") == 0) {
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
        //char *nome = (char *) malloc (100 * sizeof(char)); 
        //strncpy(nome,argv[2],(strlen(argv[2])-4));
        //strcat(nome,"_new.jpg");
        
        cout << nome;
        imwrite(nome,nova_imagem);
        cout << "fim" << endl;
        //free(nome);
    }
    
    return 0;
}
