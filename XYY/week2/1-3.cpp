#include<bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	int cnt1 = n * 2 - 1, cnt2 = 0;
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) {
			if (j == 1 || j == cnt1) cout << '*';
			else cout << ' ';
		} 
		cnt2 ++;
		cnt1 -= 2;
		cout << '\n';
	}
	cnt1 = 3, cnt2 = n - 2;
	for (int i = 1; i <= n - 1; i ++) {
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) {
			if (j == 1 || j == cnt1) cout << '*';
			else cout << ' ';
		} 
		cnt1 += 2;
		cnt2 --;
		cout << '\n';
	}
	return 0;
}
