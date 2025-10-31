#include<bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	int cnt1 = n - 1, cnt2 = 1;
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= cnt1; j ++) cout << ' ';
		for (int j = 1; j <= cnt2; j ++) cout << '*';
		cnt1 --;
		cnt2 += 2;
		cout << '\n';
	}
	cnt1 = 1, cnt2 = (n - 1) *2 - 1;
	for (int i = 1; i <= n - 1; i ++) {
		for (int j = 1; j <= cnt1; j ++) cout << ' ';
		for (int j = 1; j <= cnt2; j ++) cout << '*';
		cnt1 ++;
		cnt2 -= 2;
		cout << '\n';
	}
	return 0;
}
