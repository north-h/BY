#include<bits/stdc++.h>

using namespace std;

int main() {
	int n; cin >> n;
	char op = 'a', o1 = op + 1, o2 = op + n;
	for (int i = 1; i <= n; i ++) {
		if (i == 1 || i == n) {
			for (int j = 1; j <= n; j ++) {
				cout << op;
				op ++;
			}
			op --;
		} else {
			for (int j = 1; j <= n; j ++) {
				if (j == 1) {
					cout << o1;
					o1 ++;
				} else if (j == n) {
					cout << o2;
					o2 ++;					
				} else cout << ' ';
			}
		}
		cout << '\n';
	}
	return 0;
}


