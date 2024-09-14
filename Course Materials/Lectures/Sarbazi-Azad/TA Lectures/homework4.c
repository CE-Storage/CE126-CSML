#include <stdio.h>

int gcd(int, int);
int lcm(int, int);

int main() {
    int n;
    scanf("%d", &n);

    int a, b;
    for (int i = 0; i < n; ++i) {
        scanf("%d %d", &a, &b);
        printf("%d %d\n", gcd(a, b), lcm(a, b));
    }

    return 0;
}