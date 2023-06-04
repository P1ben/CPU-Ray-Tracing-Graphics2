#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

const float epsilon = 0.0001f;
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};
class Cone;

struct Light {
	vec3 pos;
	vec3 Le;
	Cone* cone;
	Light(vec3 _pos, vec3 _Le) {
		pos = _pos;
		Le = _Le;
		cone = nullptr;
	}
	void SetCone(Cone* c) {
		cone = c;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

struct Face {
	int x;
	int y;
	int z;
};

class Cone : public Intersectable {
private:
	vec3 pos;
	vec3 dir;
	float size;
	float alpha;
	Light* coneLight;
public:
	Cone(Material* _material, vec3 _pos, vec3 _dir, float _size, float _alpha) {
		material = _material;
		pos = _pos;
		size = _size;
		alpha = _alpha;
		dir = _dir;
		coneLight = nullptr;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 n, d, s, p, r;
		float alpha;
		n = normalize(this->dir);
		d = normalize(ray.dir);
		s = ray.start;
		p = this->pos;
		alpha = this->alpha;

		float a = (dot(n, d) * dot(n, d)) - dot(d, d) * cosf(alpha) * cosf(alpha);
		float b =  -2 * dot(d, (s - p)) * cosf(alpha) * cosf(alpha) + 2 * dot(n, d) * dot(n, s - p);
		float c = (dot(n, s - p) * dot(n, s - p)) - dot(s - p, s - p) * cosf(alpha) * cosf(alpha);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		float sel_t = (t1 > 0) ? t1 : t2;
		r = ray.start + ray.dir * sel_t;
		if (dot(r - p, n) > this->size || dot(r - p, n) < 0) {
			sel_t = t2;
		}
		r = ray.start + ray.dir * sel_t;
		if (dot(r - p, n) > this->size || dot(r - p, n) < 0) {
			return hit;
		}

		hit.t = sel_t;
		hit.position = ray.start + ray.dir * hit.t;
		r = hit.position;
		hit.normal = normalize(2 * dot(r - p, n) * n - 2 * (r - p) * cosf(alpha) * cosf(alpha));
		hit.material = material;
		return hit;
	}
	void SetPos(vec3 pos, vec3 dir) {
		this->pos = pos;
		this->dir = dir;
		if (coneLight != nullptr) {
			coneLight->pos = (pos + this->dir * 0.01);
		}
	}
	vec3 GetPos() {
		return pos;
	}
	void AddConeLight(Light* cl) {
		coneLight = cl;
		coneLight->pos = (pos + this->dir * 0.01);
		coneLight->cone = this;
	}
	Light* GetConeLight() {
		return coneLight;
	}
	vec3 GetNormal() {
		return normalize(dir);
	}
	float GetAlpha() {
		return alpha;
	}
};

class Platonic : public Intersectable{
private:
	vec3 pos;
	bool only_inside;
	float size;
	std::vector<vec3> points;
	std::vector<Face> faces;
public:
	Platonic(Material* _material, vec3 _pos, float _size, bool _only_inside) {
		pos = _pos;
		size = _size;
		only_inside = _only_inside;
		points = std::vector<vec3>();
		faces = std::vector<Face>();
		material = _material;
	}
	void AddPoint(vec3 point) {
		points.push_back(point * size + pos);
	}
	void AddFace(Face face) {
		faces.push_back(face);
	}
	void Init(const char* inp) {
		std::vector<char*> lines = std::vector<char*>();
		int pos = 0;
		int string_pos = 0;
		bool new_line = true;
		while (inp[pos] != '\0') {
			if (new_line) {
				lines.push_back(new char[50]);
				new_line = false;
			}
			if (inp[pos] == '\n') {
				lines[lines.size() - 1][string_pos] = '\0';
				string_pos = 0;
				new_line = true;
			}
			else {
				lines[lines.size() - 1][string_pos] = inp[pos];
				string_pos++;
			}
			pos++;
		}
		if (string_pos != 0) {
			lines[lines.size() - 1][string_pos] = '\0';
		}
		for (char* ch : lines) {
			if (ch[0] == 'v') {
				const char* temp;
				temp = ch;
				float a = 0;
				float b = 0;
				float c = 0;
				char t[10];
				sscanf(temp, "%s %f %f %f", t, &a, &b, &c);
				this->AddPoint(vec3(a, b, c));
			}
			else if (ch[0] == 'f') {
				const char* temp;
				temp = ch;
				int a = 0;
				int b = 0;
				int c = 0;
				char t[10];
				sscanf(temp, "%s %d %d %d", t, &a, &b, &c);
				this->AddFace({a, b, c});
			}
		}
		for (int i = 0; i < lines.size(); i++) {
			delete[] lines[i];
		}
	}
	Hit intersect(const Ray& ray) {
		Hit hit;

		std::vector<Hit> hits = std::vector<Hit>();
		vec3 r1, r2, r3, p, n;
		float alpha, beta, gamma, t;
		for (int i = 0; i < faces.size(); i++) {
			r1 = points[faces[i].x - 1];
			r2 = points[faces[i].y - 1];
			r3 = points[faces[i].z - 1];

			n = normalize(cross(r2 - r1, r3 - r1));
			t = dot((r2 - ray.start), n) / dot(ray.dir, n);
			p = ray.start + ray.dir * t;

			vec3 in_pointing = pos - r2;


			if ((dot(cross(r2 - r1, p - r1), n) > 0 && (dot(cross(r3 - r2, p - r2), n)) > 0 && (dot(cross(r1 - r3, p - r3), n) > 0))) {
				Hit temp;
				temp.t = t;
				temp.position = p;
				temp.material = material;
				if (dot(n, in_pointing) <= 0) { n = -n; }
				temp.normal = n;
				if (only_inside) {
					if (dot(n, ray.dir) <= 0.0f) {
						hits.push_back(temp);
						break;
					}
				}
				else {
					hits.push_back(temp);
				}
				if (hits.size() == 2) break;
			}
		}

		if (hits.size() == 1) {
			return hits[0];
		}
		else if (hits.size() > 1) {
			float closest_t = 10000000000.0f;
			int closest_i = 0;
			for (int j = 0; j < hits.size(); j++) {
				if (hits[j].t < closest_t) {
					closest_t = hits[j].t;
					closest_i = j;
				}
			}
			return hits[closest_i];
		}
		return hit;
	}
};
vec3 cam_pos;

// Src: https://people.sc.fsu.edu/~jburkardt/data/obj/dodecahedron.obj
void FillWithDodecahedron(Platonic* pt) {
	pt->Init(R"(v  -0.57735  -0.57735  0.57735
v  0.934172  0.356822  0
v  0.934172  -0.356822  0
v  -0.934172  0.356822  0
v  -0.934172  -0.356822  0
v  0  0.934172  0.356822
v  0  0.934172  -0.356822
v  0.356822  0  -0.934172
v  -0.356822  0  -0.934172
v  0  -0.934172  -0.356822
v  0  -0.934172  0.356822
v  0.356822  0  0.934172
v  -0.356822  0  0.934172
v  0.57735  0.57735  -0.57735
v  0.57735  0.57735  0.57735
v  -0.57735  0.57735  -0.57735
v  -0.57735  0.57735  0.57735
v  0.57735  -0.57735  -0.57735
v  0.57735  -0.57735  0.57735
v  -0.57735  -0.57735  -0.57735

f  19  3  2
f  12  19  2
f  15  12  2
f  8  14  2
f  18  8  2
f  3  18  2
f  20  5  4
f  9  20  4
f  16  9  4
f  13  17  4
f  1  13  4
f  5  1  4
f  7  16  4
f  6  7  4
f  17  6  4
f  6  15  2
f  7  6  2
f  14  7  2
f  10  18  3
f  11  10  3
f  19  11  3
f  11  1  5
f  10  11  5
f  20  10  5
f  20  9  8
f  10  20  8
f  18  10  8
f  9  16  7
f  8  9  7
f  14  8  7
f  12  15  6
f  13  12  6
f  17  13  6
f  13  1  11
f  12  13  11
f  19  12  11)");
}

// Src: https://people.sc.fsu.edu/~jburkardt/data/obj/icosahedron.obj
void FillWithIcosahedron(Platonic* pt) {
	pt->AddPoint(vec3(0, -0.525731, 0.850651));
	pt->AddPoint(vec3(0.850651, 0, 0.525731));
	pt->AddPoint(vec3(0.850651, 0, -0.525731));
	pt->AddPoint(vec3(-0.850651, 0, -0.525731));
	pt->AddPoint(vec3(-0.850651, 0, 0.525731));
	pt->AddPoint(vec3(-0.525731, 0.850651, 0));
	pt->AddPoint(vec3(0.525731, 0.850651, 0));
	pt->AddPoint(vec3(0.525731, -0.850651, 0));
	pt->AddPoint(vec3(-0.525731, -0.850651, 0));
	pt->AddPoint(vec3(0, -0.525731, -0.850651));
	pt->AddPoint(vec3(0, 0.525731, -0.850651));
	pt->AddPoint(vec3(0, 0.525731, 0.850651));

	pt->AddFace({ 2, 3, 7 });
	pt->AddFace({ 2, 8, 3 });
	pt->AddFace({ 4, 5, 6 });
	pt->AddFace({ 5, 4, 9 });
	pt->AddFace({ 7, 6, 12 });
	pt->AddFace({ 6, 7, 11 });
	pt->AddFace({ 10, 11, 3 });
	pt->AddFace({ 11, 10, 4 });
	pt->AddFace({ 8, 9, 10 });
	pt->AddFace({ 9, 8, 1 });
	pt->AddFace({ 12, 1, 2 });
	pt->AddFace({ 1, 12, 5 });
	pt->AddFace({ 7, 3, 11 });
	pt->AddFace({ 2, 7, 12 });
	pt->AddFace({ 4, 6, 11 });
	pt->AddFace({ 6, 5, 12 });
	pt->AddFace({ 3, 8, 10 });
	pt->AddFace({ 8, 2, 1 });
	pt->AddFace({ 4, 10, 9 });
	pt->AddFace({ 5, 9, 1 });
}
void FillWithCube(Platonic* pt) {
	pt->AddPoint(vec3(-1, -1, -1));
	pt->AddPoint(vec3(-1, -1, 1));
	pt->AddPoint(vec3(-1, 1, -1));
	pt->AddPoint(vec3(-1, 1, 1));
	pt->AddPoint(vec3(1, -1, -1));
	pt->AddPoint(vec3(1, -1, 1));
	pt->AddPoint(vec3(1, 1, -1));
	pt->AddPoint(vec3(1, 1, 1));

	pt->AddFace({ 1, 5, 7 });
	pt->AddFace({ 1, 3, 7 });
	pt->AddFace({ 1, 2, 3 });
	pt->AddFace({ 1, 5, 2 });
	pt->AddFace({ 6, 5, 2 });
	pt->AddFace({ 6, 5, 7 });
	pt->AddFace({ 6, 8, 7 });
	pt->AddFace({ 6, 2, 4 });
	pt->AddFace({ 4, 3, 2 });
	pt->AddFace({ 4, 8, 6 });
	pt->AddFace({ 4, 8, 7 });
	pt->AddFace({ 4, 3, 7 });
}

Platonic* dodeca;
Platonic* icosa;
Platonic* cube;
std::vector<Cone*> cones;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		objects.clear();
		lights.clear();
		vec3 eye = cam_pos, vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.0f, 0.0f, 0.0f);
		objects.push_back(icosa);
		objects.push_back(cube);
		objects.push_back(dodeca);
		for (Cone* c : cones) {
			objects.push_back(c);
			if(c->GetConeLight() != nullptr)
				lights.push_back(c->GetConeLight());
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = vec3(1, 1, 1) * 0.2f * (1.0f + dot(normalize(hit.normal), -ray.dir));
		for (Light* light : lights) {
			vec3 r = hit.position - light->cone->GetPos();
			float angle = acosf(dot(normalize(light->cone->GetNormal()), normalize(r)));
			if (angle <= light->cone->GetAlpha() + 0.02) {
				Hit l = firstIntersect(Ray(hit.position + hit.normal * epsilon, normalize(light->pos - hit.position)));
				if (l.t < 0 || l.t > length(light->pos - hit.position)) {
					outRadiance = outRadiance + 0.8f * light->Le * 1.0f / (length(hit.position - light->pos) * length(hit.position - light->pos));
				}
			}
		}
		return outRadiance;
	}
	Camera GetCamera() {
		return camera;
	}
};


class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

Scene scene;
FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	cam_pos = vec3(-2.5, 0, 2.5);
	vec3 kd(0.4f, 0.4f, 0.4f), ks(/*0.4f, 0.4f, 0.4f*/2, 2, 2);
	Material* material = new Material(kd, ks, 100);
	kd = vec3(0.4f, 0.4f, 0.4f); ks = vec3(/*0.4f, 0.4f, 0.4f*/2, 2, 2);
	Material* material_cone = new Material(kd, ks, 100);

	icosa = new Platonic(material, vec3(-0.4, -0.55f, -0.4), 0.5f, false);
	cube = new Platonic(material, vec3(0, 0.0f, 0), 1.0f, true);
	dodeca = new Platonic(material, vec3(0.3, -0.7f, 0.4), 0.3f, false);

	cones.push_back(new Cone(material_cone, vec3(0, 1.0f, 0), vec3(0, -1, 0), 0.2f, M_PI / 8.0f));
	cones[0]->AddConeLight(new Light(vec3(0, 0, 0), vec3(1, 0, 0)));
	cones.push_back(new Cone(material_cone, vec3(0.8, -0.2, -1), vec3(0, 0, 1), 0.2, M_PI / 8.0f));
	cones[1]->AddConeLight(new Light(vec3(0, 0, 0), vec3(0, 1, 0)));
	cones.push_back(new Cone(material_cone, vec3(1, -0.7f, -0.4), vec3(-1, 0, 0), 0.2, M_PI / 8.0f));
	cones[2]->AddConeLight(new Light(vec3(0, 0, 0), vec3(0, 0, 1)));

	FillWithIcosahedron(icosa);
	FillWithCube(cube);
	FillWithDodecahedron(dodeca);

	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

long last_render = 0;
// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}
// Window has become invalid: Redraw

// Key of ASCII code pressed
bool pressed[256] = { false, };
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	pressed[key] = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	pressed[key] = false;
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}
bool down = true;
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	pY = windowHeight - pY;
	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; down = false; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:
	{
		if (down != false) break;
		down = true;
		Hit first = scene.firstIntersect(Ray(scene.GetCamera().getRay(pX, pY)));
		if (first.t > 0) {
			Cone* closest = nullptr;
			float smallest_dist = 100000000.0f;
			for (Cone* c : cones) {
				if (length(c->GetPos() - first.position) < smallest_dist) {
					closest = c;
					smallest_dist = length(c->GetPos() - first.position);
				}
			}
			if (closest != nullptr) {
				closest->SetPos(first.position, normalize(first.normal));
				scene.build();
				std::vector<vec4> image(windowWidth * windowHeight);
				scene.render(image);
				delete fullScreenTexturedQuad;
				fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
				glutPostRedisplay();
			
			}
		}
	}
		break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}


// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	
}
