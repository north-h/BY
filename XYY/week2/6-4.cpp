#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

int get(int x) {
	int sum = 0;
	while (x) {
		if (x % 7 == 5) sum ++;
		x /= 7;
	}
	return sum;
}

bool cmp(int a, int b) {
	if (get(a) == get(b)) return a > b;
	return get(a) > get(b);
}

int main() {
	int n; cin >> n;
	for (int i = 0; i < n; i ++) cin >> a[i];
	sort(a, a + n, cmp);
	for (int i = 0; i < n; i ++) cout << a[i] << '\n';
	return 0;
}


