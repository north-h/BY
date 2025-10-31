#include<bits/stdc++.h>

using namespace std;

int a[1000010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int x = 0;
	for (int i = 0; ; i ++) {
		cin >> a[i];
		if (a[i] == 0) break;
		x ++;	
	}
	int ans = 100000001;
	sort(a, a + x);
	for (int i = 1; i < x; i ++) {
		int t = a[i] - a[i - 1];
		if (t < ans) ans = t;
	}
	cout << ans << '\n';
	return 0;
}


