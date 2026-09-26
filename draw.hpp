#pragma once
#include "matrix/matrix.hpp"
#include <windows.h>
#include <conio.h>

#define D "██"
#define N "□□"

void draw(int img[][28]){
    system("cls");
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            if (img[i][j] == 0){
                std::cout << N;
            }else{
                std::cout << D;
            }
        }
        std::cout << "\n";
    }
}

Matrix GetImage(){
    SetConsoleOutputCP(65001);
    Matrix res(28, 28);
    int canvas[28][28]; memset(canvas, 0, sizeof(canvas));
    int cur[2] = {14, 14};
    canvas[14][14] = 2;
    draw(canvas);

    while (true){
        int ch = _getch();
        if (ch == 0 || ch == 0xE0){
            ch = _getch();

            switch(ch){
                case 75:
                    if (cur[1] > 0) --cur[1];
                    if (canvas[cur[0]][cur[1] + 1] != 1) canvas[cur[0]][cur[1] + 1] = 0;
                    break;
                case 77:
                    if (cur[1] < 27) ++cur[1];
                    if (canvas[cur[0]][cur[1] - 1] != 1) canvas[cur[0]][cur[1] - 1] = 0;
                    break;
                case 72:
                    if (cur[0] > 0) --cur[0];
                    if (canvas[cur[0] + 1][cur[1]] != 1) canvas[cur[0] + 1][cur[1]] = 0;
                    break;
                case 80:
                    if (cur[0] < 27) ++cur[0];
                    if (canvas[cur[0] - 1][cur[1]] != 1) canvas[cur[0] - 1][cur[1]] = 0;
                    break;
            }
            if (canvas[cur[0]][cur[1]] != 1) canvas[cur[0]][cur[1]] = 2;
            draw(canvas);
        }else{
            if (ch == 's'){
                canvas[cur[0]][cur[1]] = 1;
                res.set({cur[0], cur[1]}, 1);
            }else if (ch == 'r'){
                return res;
            }
        }
    }
}