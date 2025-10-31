#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n, k;
	cin >> n >> k;
	for (int i = 0; i < k; i ++) cin >> a[i];
	for (int i = 0; i < n - k; i ++) cin >> b[i];
	sort(a, a + k);
	sort(b, b + n - k, cmp);
	for (int i = 0; i < k; i ++) cout << a[i] << ' ';
	for (int i = 0; i < n - k; i ++) cout << b[i] << ' ';
	return 0;
}


