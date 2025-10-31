#include<bits/stdc++.h>
using namespace std;
int main() {
	int n; cin >> n;
	int cnt1 = n * 2 + 1, cnt2 = 0;
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= cnt1; j ++) cout << '*';
		cout << '\n';
		cnt1 -= 2;
	}
	return 0;
}
