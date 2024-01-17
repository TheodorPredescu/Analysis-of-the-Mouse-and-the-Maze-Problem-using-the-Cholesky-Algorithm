#include <iostream>
#include <vector>

using namespace std;
int main() {


    printf("Introduceti nr:");
    int nr;
    cin>>nr;
    
    vector<vector<float>> matrix(nr, vector<float>(nr, 0.0));

    int x =1, sum_linii_precedente = 0, nr_descendenti, elemente_linie;
    bool ok = false;

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nr; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout<<"************************\n";
    while (ok == false) {
        for (int i = 1 ; i <= x; i ++) {
            nr_descendenti = 0;
            if (i + sum_linii_precedente <= nr) {
                // cout<<i + sum_linii_precedente<< " -> descendent1: "<<i + sum_linii_precedente + x<<endl;
                if (nr >= i + sum_linii_precedente + x) {
                    nr_descendenti++;
                }
                if (nr >= i + sum_linii_precedente + x + 1) {
                    nr_descendenti++;
                }
            } else {
                ok = true;
                break;
            }
            if (nr_descendenti != 0) {
                if (nr_descendenti >= 1) {
                    matrix[i + sum_linii_precedente - 1][ i + sum_linii_precedente - 1 + x] = 1;
                    matrix[i + sum_linii_precedente - 1 + x][i + sum_linii_precedente - 1] = 1;
                }
                if (nr_descendenti >= 2) {
                    matrix[i + sum_linii_precedente - 1][i + sum_linii_precedente - 1 + x + 1] = 1;
                    matrix[i + sum_linii_precedente - 1 + x + 1][i + sum_linii_precedente - 1] = 1;
                }
            }

        }
        sum_linii_precedente += x;
        x++;
    }

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nr; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}