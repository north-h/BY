#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010], c[1000010];

bool get1(int x) {
	int sum = 0;
	for (int i = 1; i < x; i ++) {
		if (x % i == 0) sum += i;
	}
	return sum == x;
}

bool get2(int x) {
	int sum = 0;
	int y = x;
	while (y) {
		sum += pow(y % 10, 3);
		y /= 10;
	}
	return sum == x;
}

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n; cin >> n;
	int aa = 0, bb = 0, cc = 0;
	for (int i = 0; i < n; i ++) {
		int x; cin >> x;
		if (get1(x)) a[aa] = x, aa ++;
		else if (get2(x)) b[bb] = x, bb ++;
		else c[cc] = x, cc ++;
	}
	sort(a, a + aa, cmp);
	sort(b, b + bb);
	sort(c, c + cc);
	for (int i = 0; i < aa; i ++) cout << a[i] << ' ';
	for (int i = 0; i < cc; i ++) cout << c[i] << ' ';
	for (int i = 0; i < bb; i ++) cout << b[i] << ' ';
	return 0;
}


