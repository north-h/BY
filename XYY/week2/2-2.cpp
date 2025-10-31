#include<bits/stdc++.h>

using namespace std;

int main() {
	char op, oo = 'A';
	cin >> op;
	int n = op - oo + 1, cnt1 = 1, cnt2 = n - 1;
//	cout << n << '\n';
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) {
			if (j == 1 || j == cnt1 || i == n) cout << oo;
			else cout << ' ';
		}
		cnt2 --;
		cnt1 += 2;
		oo ++;
		cout << '\n';
	}
	return 0;
}


