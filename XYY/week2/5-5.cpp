#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

bool cmp(int a, int b) {
	return a > b;
}

int main() {
	int n, m;
	cin >> n >> m;
	for (int i = 0; i < n; i ++) cin >> a[i];
	sort(a, a + n);
	int ans = 0;
	for (int i = 0; i < n; i ++) {
		if (m >= a[i]) m -= a[i];
		else break;
		ans ++; 
	}
	cout << ans << '\n';
	return 0;
}


