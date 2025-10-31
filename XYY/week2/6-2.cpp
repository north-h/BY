#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n; cin >> n;
	int aa = 0, bb = 0;
	for (int i = 0; i < n; i ++) {
		int x;
		cin >> x;
		if (x > 0) a[aa] = x, aa ++;
		else b[bb] = x, bb ++;
	}
	sort(b, b + bb);
	sort(a, a + aa, cmp);
	for (int i = 0; i < bb; i ++) cout << b[i] << ' ';
	for (int i = 0; i < aa; i ++) cout << a[i] << ' ';
	return 0;
}


