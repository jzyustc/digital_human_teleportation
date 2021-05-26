#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#define pos(u, v) (((u)*(w))+v)

void
quick_render(float image[][3], float *depthmap, float *mask, float point_list[][6], int list_num, float k, float f,
             float image_range[2][2], int h, int w, int point_size) {

    int size = point_size / 2;

    int u, v, ut, vt;
    float depth;

    for (int pid = 0; pid < list_num; pid++) {

        depth = point_list[pid][2];

        u = round(point_list[pid][0] / depth * k * f - image_range[0][0]);
        if (u < 0 || u >= h)continue;

        v = round(point_list[pid][1] / depth * k * f - image_range[1][0]);
        if (v < 0 || v >= w)continue;

        for (int i = -size; i < size + 1; ++i) {
            ut = u + i;
            if (ut < 0 || ut >= h)continue;
            for (int j = -size; j < size + 1; ++j) {
                vt = v + j;
                if (vt < 0 || vt >= w)continue;

                if (mask[pos(ut, vt)] == 0. || depthmap[pos(ut, vt)] < depth) {
                    image[pos(ut, vt)][0] = point_list[pid][3];
                    image[pos(ut, vt)][1] = point_list[pid][4];
                    image[pos(ut, vt)][2] = point_list[pid][5];
                    depthmap[pos(ut, vt)] = depth;
                    mask[pos(ut, vt)] = 1;
                }
            }
        }
    }
}

void number_s_get(char *number_s, char *line) {
    char *p = number_s;
    char *q = line;
    int i = 15;
    for (; line[i] != 0; ++i) {
        p[i - 15] = line[i];
    }
    p[i - 15] = 0;
}

void read(char *path, float point_list[][6]) {
    long i = 0;

    char lists[6][100];
    char alpha[100];

    char read[100];
    FILE *fpread;
    fpread = fopen(path, "r");

    char *temp;
    char number_s[20];
    long number = 0;
    while (1) {
        fgets(read, 100, fpread);
        if (strncmp(read, "end_header", 10) == 0) break;

        if (strncmp(read, "element vertex ", 15) == 0) {
            number_s_get(number_s, read);
            number = strtol(number_s, &temp, 10);
        }
    }


    for (i = 0; i < number; ++i) {
        fscanf(fpread, "%s %s %s %s %s %s %s\n", lists[0], lists[1], lists[2], lists[3], lists[4], lists[5], alpha);
        point_list[i][0] = -strtof(lists[1], &temp);
        point_list[i][1] = strtof(lists[0], &temp);
        point_list[i][2] = strtof(lists[2], &temp);
        point_list[i][3] = strtof(lists[3], &temp);
        point_list[i][4] = strtof(lists[4], &temp);
        point_list[i][5] = strtof(lists[5], &temp);
    }

}
