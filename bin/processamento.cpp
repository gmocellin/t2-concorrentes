#include <cstdio>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "processamento.h"

// para openCV
#define SCALARBLUE 0
#define SCALARGREEN 1
#define SCALARRED 2
// para openMP e openMPI
#define MASTER 0
#define RED 1
#define BLUE 2
#define GREEN 3

using namespace std;
using namespace cv;

void sum5(Mat *matriz, int linha, int coluna, int flag) {
    int i,j;
    unsigned char sum = 0;
    Scalar temp;
    for(i = (linha - 2); i <= (linha + 2); i++){
        for(j = (coluna - 2); j <= (coluna + 2); j++) {
            if (i >= 0 && j >= 0) {
                temp = (*matriz).at<unsigned char>(i,j);
                sum += temp[flag];
            }
        }
    }
    temp[flag] = (sum/25);
    (*matriz).at<Scalar>(linha,coluna) = temp;
}

processamento::processamento(Mat *matriz) {
    this->setMatriz(matriz);
    this->setRGB();
}

processamento::~processamento() {
    this->matriz = NULL;
    this->R = NULL;
    this->G = NULL;
    this->B = NULL;
}

void processamento::setMatriz(Mat *matriz) {
    this->matriz = matriz;
}

void processamento::setRGB() {
    vector<Mat> aux;
    split((*matriz), aux);
    this->B = &aux[0];
    this->G = &aux[1];
    this->R = &aux[2];
}

void processamento::smooth(Mat *matriz, int flag) {
    int i,j;
    for (i = 0; i < matriz->rows; i++) {
        for (j = 0; j < matriz->cols; j++) {
            sum5(matriz,i,j,flag);
        }
    }
}

Mat* processamento::getMatriz() {
    return this->matriz;
}

Mat* processamento::getR() {
    return this->R;
}

Mat* processamento::getG() {
    return this->G;
}

Mat* processamento::getB() {
    return this->B;
}

Mat processamento::processando() {}

paralelo::paralelo(Mat *matriz) : processamento(matriz) {}

paralelo::~paralelo() {
    matriz = NULL;
    R = NULL;
    G = NULL;
    B = NULL;
}

Mat paralelo::processando() {
}

sequencial::sequencial(Mat *matriz) : processamento(matriz) {}

sequencial::~sequencial() {
    matriz = NULL;
    R = NULL;
    G = NULL;
    B = NULL;
}

Mat sequencial::processando() {
    smooth(R, SCALARRED);
    smooth(G, SCALARGREEN);
    smooth(B, SCALARBLUE);
    
    Mat aux[3], R_imagem;
    aux[0] = *(R);
    aux[1] = *(G);
    aux[2] = *(B);
    
    merge(aux,3,R_imagem);
    
    return R_imagem;
}