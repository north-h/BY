#include<bits/stdc++.h>

using namespace std;

int main() {
	int n; cin >> n;
	int sum = 0;
	for (int i = 2; i <= n; i ++) {
		bool ok = true;
		for (int j = 2; j < i; j ++) {
			if (i % j == 0) {
				ok = false;
				break;
			}
		}
		for (int j = 2; j < n - i; j ++) {
			if ((n - i) % j == 0) {
				ok = false;
				break;
			}
		}
		if (ok) {
			cout << i << ' ' << n - i << '\n';
			return 0;
		}
	}
	cout << sum << '\n';
	return 0;
}
