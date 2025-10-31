#include<bits/stdc++.h>

using namespace std;

int a[1010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n, k;
	cin >> n >> k;
	for (int i = 0; i < n; i ++) cin >> a[i];
	sort(a, a + n, cmp);
//	for (int i = 0; i < n; i ++) cout << a[i] << ' ';
	cout << a[k - 1] << '\n';
	return 0;
}


