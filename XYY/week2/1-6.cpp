#include<bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	char op = 'A';
	int cnt1 = n * 2 + 1, cnt2 = 0;
	for (int i = 1; i <= n + 1; i ++) {
		char t = op;
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) {
			cout << t;
			t ++;
		}
		cnt1 -= 2;
		cnt2 ++;
		cout << '\n';
		op ++;
	}
	op -= 2;
	cnt1 = 3, cnt2 = n - 1;
	for (int i = 1; i <= n; i ++) {
		char t = op;
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) {
			cout << t;
			t ++;
		}
		cnt1 += 2;
		cnt2 --;
		cout << '\n';
		op --;
	}
	return 0;
}


