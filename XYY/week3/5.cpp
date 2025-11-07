#include<bits/stdc++.h>

using namespace std;

int main() {
	int n; cin >> n;
	int sum = 0;
	bool f = true;
	int x = n;
	while (n) {
		bool ok = true;
		for (int j = 2; j < n; j ++) {
			if (n % j == 0) {
				ok = false;
				break;
			}
		}
		if (!ok || n <= 1) f = false;
		n /= 10;
	}
	if (f) cout << "yes" << '\n';
	else cout << "no" << '\n';
	return 0;
}
