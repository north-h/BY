#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n, l1, r1, l2, r2;
	cin >> n >> l1 >> r1 >> l2 >> r2;
	for (int i = 1; i <= n; i ++) cin >> a[i];
	sort(a + l1, a + r1 + 1);
	sort(a + l2, a + r2 + 1, cmp);
	for (int i = 1; i <= n; i ++) cout << a[i] << ' ';
	return 0;
}


