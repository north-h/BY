#include<bits/stdc++.h>

using namespace std;

int main() {
	int n, m, x;
	char op;
	cin >> n >> m >> op >> x;
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= m; j ++) {
			if (j == 1 || j == m || i ==1 || i == n || x == 1) cout << op;
			else cout << ' ';
		}
		cout << '\n';
	}
	return 0;
}


