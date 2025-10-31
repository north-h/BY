#include<bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	int cnt1 = n * 2 - 1, cnt2 = 1;
	for (int i = 1; i <= n + 1; i ++) {
//		cout << cnt1 << ' ' << cnt2 << '\n';
		if (i == 1) {
			for (int j = 1; j <= n * 2 + 1; j ++) cout << '*';
		}
		else {
			for (int j = 1; j <= cnt2; j ++) cout << ' ';
			for (int j = 1; j <= cnt1; j ++) {
				if (j == 1 || j == cnt1) cout << '*';
				else cout << ' ';
			} 
			cnt2 ++;
			cnt1 -= 2;
		}
		cout << '\n';
	}
//	cout << cnt1 << ' ' << cnt2 << '\n';
	cnt1 = 3, cnt2 = n - 1;
	for (int i = 1; i <= n; i ++) {
		if (i == n) {
			for (int j = 1; j <= n * 2 + 1; j ++) cout << '*';
		}
		else {
			for (int j = 1; j <= cnt2; j ++) cout << ' ';
			for (int j = 1; j <= cnt1; j ++) {
				if (j == 1 || j == cnt1) cout << '*';
				else cout << ' ';
			} 
			cnt2 --;
			cnt1 += 2;
		}
		cout << '\n';
	}
	return 0;
}
