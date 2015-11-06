#include <iostream>
#include <cstring>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <mpi.h>
#include <omp.h>

#define NTHREADS 4

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
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
    
    imagem.copyTo(nova_imagem);

    int i,j, linha, coluna;
    
    if (strcmp (argv[1], "thread") == 0) {
        for (i = 0; i < imagem.rows; i++) {
            for (j = 0; j < imagem.cols; j++) {
                float sum[3] = {0, 0, 0};
                //omp_set_num_threads(NTHREADS);
                // #pragma omp parallel for //shared(i,j, linha, coluna, sum)
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
     
    return 0;
}
